"""Independent, image-only advisory reviews. Never admits data or supplies human MOS."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from PIL import Image
from urbanphotomeshqa.integrity import sha256_file


PROMPT = '''Assess the visual delivery quality of ONE textured photogrammetric building mesh.
You receive fixed views of the same building, at their labeled resolutions. These are observations,
not ground truth. Judge this asset independently; no reference clean building is available.
Grade the observable overall quality: 5 excellent, stable outline/structure/texture clarity and
alignment with no readily noticeable visual defect at normal viewing distance; 4 good, noticeable
but small/mild defects without substantial loss of usability; 3 fair, clearly visible defects in
structure, texture clarity or alignment, still useful but requires repair; 2 poor, severe defects
that prevent normal delivery, though the building remains recognizable; 1 very poor, extensive
damage, loss of core structure or severely failed texture, only broad identity remains useful.
Do not assume that an original-looking image deserves 5. Do not grade photographic aesthetics.
Natural shadows, reflected surroundings in glass, paint colors, intentional plain walls, vegetation,
and actual architectural asymmetry are not automatically defects. Do not invent defects behind
unseen surfaces or missing fine geometry without evidence. Point to numbered views for defects.
Report visible attributes separately: G1 geometry missing/holes; G2 shape distortion/spikes/warping;
G3 geometric detail loss/oversmoothing; T1 texture blur/resolution loss; T2 mapping misalignment,
stretching or ghosting; T3 missing/invalid texture or implausible radiometric inconsistency.
Each value is 1 for a supported visible defect, 0 for no visible defect in the supplied evidence,
null if ambiguous. Different defects may coexist. Texture detail loss is not automatically G3.
Return only a JSON object with keys: overall_scale (integer 1..5 or null), uncertain (boolean),
visible_attributes (object containing G1,G2,G3,T1,T2,T3), observations (array of at most 4 short
strings naming view numbers and concrete evidence), plausibility_concern (string or null).
Use uncertainty if the image evidence cannot distinguish an artifact from a real building feature.
Never infer a requested grade, generation recipe, parameter, or attack label.'''


def main():
    p = argparse.ArgumentParser()
    for name in ('queue', 'model', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--limit', type=int, default=12)
    p.add_argument('--max-new-tokens', type=int, default=384)
    args = p.parse_args()
    if not 1 <= args.limit <= 240:
        raise ValueError('Bounded review run required')
    args.output.mkdir(parents=True, exist_ok=True)
    queue = json.loads(args.queue.read_text())
    # This queue contains neutral IDs and evidence folders only, never private labels.
    if any(set(row) != {'review_id', 'evidence_folders'} for row in queue):
        raise ValueError('Review queue must contain only neutral image references')
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor, AutoConfig
    torch.manual_seed(2026)
    model_type = AutoConfig.from_pretrained(args.model, local_files_only=True).model_type
    model_class = {'qwen2_5_vl': Qwen2_5_VLForConditionalGeneration,
                   'qwen3_vl': Qwen3VLForConditionalGeneration}.get(model_type)
    if model_class is None:
        raise ValueError('Only explicitly supported local vision model families are allowed')
    provenance_path = args.output/'provenance.json'
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if provenance['prompt'] != PROMPT or provenance['model_path'] != str(args.model):
            raise ValueError('Use a new review version for prompt/model changes')
    else:
        files = sorted(p for p in args.model.iterdir() if p.suffix in ('.json', '.safetensors', '.txt'))
        provenance = {'model_id': 'Qwen/'+args.model.name, 'model_type': model_type, 'model_path': str(args.model),
            'model_source': 'official Qwen repository on ModelScope', 'model_files': {p.name: sha256_file(p) for p in files},
            'prompt': PROMPT, 'seed': 2026, 'do_sample': False, 'max_new_tokens': args.max_new_tokens,
            'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
            'protocol': 'v3_independent_vlm_advisory_v1', 'formal_admitted': False, 'human_mos': False,
            'limitation': 'Uncalibrated independent machine opinion; cannot authorize dataset admission or local quality labels'}
        with provenance_path.open('x') as f:
            json.dump(provenance, f, indent=2)
    provenance_hash = sha256_file(provenance_path)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True,
        min_pixels=256*28*28, max_pixels=1024*1024)
    model = model_class.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval()
    for row in queue[:args.limit]:
        started = time.monotonic()
        identity = row['review_id']
        if len(identity) != 20 or any(c not in '0123456789abcdef' for c in identity):
            raise ValueError('Invalid neutral review identifier')
        output = args.output/(identity+'.json')
        images, content, evidence = [], [], []
        try:
            digests = set()
            for raw in row['evidence_folders']:
                folder = Path(raw)
                receipt = json.loads((folder/'receipt.json').read_text())
                digests.add(receipt['content_digest'])
                for i in range(7):
                    path = folder/f'view{i}.png'
                    if sha256_file(path) != receipt['images'][path.name]:
                        raise ValueError('Evidence image hash mismatch')
                    image = Image.open(path).convert('RGB')
                    images.append(image)
                    content.extend([{'type': 'text', 'text': f'View {i}, native image {image.width}x{image.height}:'},
                                    {'type': 'image'}])
                evidence.append({'receipt_sha256': sha256_file(folder/'receipt.json'),
                                 'images': receipt['images'], 'size': receipt['size']})
            if len(digests) != 1:
                raise ValueError('Evidence folders do not refer to one asset')
            if output.exists():
                previous = json.loads(output.read_text())
                if previous['provenance_sha256'] != provenance_hash or previous['evidence'] != evidence:
                    raise ValueError('Existing review evidence/provenance mismatch')
                print(identity, 'reused', flush=True)
                continue
            content.append({'type': 'text', 'text': PROMPT})
            messages = [{'role': 'user', 'content': content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, padding=True, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            raw_text = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            parsed, error = None, None
            try:
                cleaned = raw_text.strip()
                if cleaned.startswith('```'):
                    cleaned = cleaned.split('\n', 1)[1].rsplit('```', 1)[0].strip()
                parsed = json.loads(cleaned)
                grade = parsed.get('overall_scale')
                if grade is not None and (type(grade) is not int or grade not in range(1, 6)):
                    raise ValueError('Invalid grade')
                attrs = parsed['visible_attributes']
                if set(attrs) != {'G1','G2','G3','T1','T2','T3'} or any(v not in (0, 1, None) for v in attrs.values()):
                    raise ValueError('Invalid attribute schema')
                if type(parsed.get('uncertain')) is not bool:
                    raise ValueError('Missing uncertainty')
            except Exception as ex:
                parsed, error = None, str(ex)
            result = {'review_id': identity, 'content_digest': next(iter(digests)),
                'provenance_sha256': provenance_hash, 'evidence': evidence, 'raw_response': raw_text,
                'opinion': parsed, 'parse_error': error, 'elapsed_seconds': time.monotonic()-started,
                'formal_admitted': False, 'human_mos': False, 'local_quality_labels': None}
            with output.open('x') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(identity, 'reviewed', round(result['elapsed_seconds'], 1), 'seconds', flush=True)
            del inputs, generated
        except Exception as ex:
            with (args.output/'failures.jsonl').open('a') as f:
                f.write(json.dumps({'review_id': identity, 'error': str(ex)})+'\n')
            print(identity, 'FAILED', str(ex), flush=True)
        finally:
            for image in images:
                image.close()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
