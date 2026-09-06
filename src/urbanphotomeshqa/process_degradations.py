"""V3 process-inspired degradations. All UV coordinates use native glTF V-down."""
from dataclasses import replace
import heapq
import numpy as np
from PIL import Image, ImageDraw
from scipy import sparse
from scipy.ndimage import map_coordinates, label, find_objects, distance_transform_edt
from .mesh_attacks import face_adjacency, recompute_vertex_normals


def topology(mesh):
    points, inverse = np.unique(np.round(mesh.vertices, 8), axis=0, return_inverse=True)
    return points, inverse, inverse[mesh.faces]


def areas(mesh):
    t = mesh.vertices[mesh.faces]
    return np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1)*.5


class SurfaceRegion:
    """One deterministic geodesic expansion; all levels are nested by surface area."""
    def __init__(self, mesh, seed, first_face=None):
        _, _, faces = topology(mesh)
        adjacency = face_adjacency(faces)
        centers = mesh.vertices[mesh.faces].mean(axis=1)
        self.area = areas(mesh)
        rng = np.random.default_rng(seed)
        first = int(rng.choice(len(faces), p=self.area/self.area.sum())) if first_face is None else int(first_face)
        seen = np.zeros(len(faces), bool)
        order = []
        queue = [(0., first)]
        while len(order) < len(faces):
            if not queue:
                # Disconnected components: explicitly bridge to nearest unvisited face.
                remaining = np.flatnonzero(~seen)
                next_face = remaining[np.argmin(np.linalg.norm(centers[remaining]-centers[first], axis=1))]
                queue = [(0., int(next_face))]
            distance, face = heapq.heappop(queue)
            if seen[face]:
                continue
            seen[face] = True
            order.append(face)
            for other in adjacency[face]:
                if not seen[other]:
                    heapq.heappush(queue, (distance+float(np.linalg.norm(centers[face]-centers[other])), other))
        self.order = np.asarray(order)
        self.cumulative = np.cumsum(self.area[self.order])

    def mask(self, fraction):
        if not 0 <= fraction <= 1:
            raise ValueError('Surface fraction must be in [0,1]')
        if fraction == 0:
            return np.zeros(len(self.order), bool)
        count = min(len(self.order), int(np.searchsorted(self.cumulative, fraction*self.area.sum()))+1)
        result = np.zeros(len(self.order), bool)
        result[self.order[:count]] = True
        return result

    def weights(self, fraction):
        """Fractional last triangle avoids a huge face overshooting a small budget."""
        if not 0 <= fraction <= 1:
            raise ValueError('Surface fraction must be in [0,1]')
        before = np.r_[0., self.cumulative[:-1]]
        values = np.clip((fraction*self.area.sum()-before)/np.maximum(self.area[self.order],1e-20),0,1)
        result = np.zeros_like(self.area)
        result[self.order] = values
        return result


class MultiSurfaceRegion:
    """Disjoint spatial territories with nested growth from dispersed anchors.

    Territories are a budgeting device, not Patch truth. Each region receives a
    share of the total surface budget; the full budget is never repeated per seed.
    """
    def __init__(self, mesh, seed, count=3):
        self.area = areas(mesh)
        centers = mesh.vertices[mesh.faces].mean(axis=1)
        valid = np.flatnonzero(self.area > 0)
        if len(valid) < count or count < 2:
            raise ValueError('Insufficient surface for multiple regions')
        rng = np.random.default_rng(seed)
        anchors = [int(rng.choice(valid, p=self.area[valid]/self.area[valid].sum()))]
        for _ in range(count-1):
            distances = np.min(np.linalg.norm(centers[valid,None]-centers[anchors],axis=2),axis=1)
            distances[np.isin(valid,anchors)] = -1
            anchors.append(int(valid[np.argmax(distances)]))
        self.anchors = anchors
        self.region_ids = np.argmin(np.linalg.norm(centers[:,None]-centers[anchors],axis=2),axis=1)
        self.orders = []
        for i, anchor in enumerate(anchors):
            order = SurfaceRegion(mesh,seed+i,anchor).order
            self.orders.append(order[self.region_ids[order] == i])

    def weights(self, fraction):
        if not 0 <= fraction <= 1:
            raise ValueError('Surface fraction must be in [0,1]')
        result = np.zeros_like(self.area)
        for order in self.orders:
            area = self.area[order]
            before = np.r_[0.,np.cumsum(area)[:-1]]
            result[order] = np.clip((fraction*area.sum()-before)/np.maximum(area,1e-20),0,1)
        return result

    def mask(self, fraction):
        # Face deletion cannot delete a fraction of a triangle; callers must
        # record actual area rather than present requested area as exact.
        return self.weights(fraction) > 0


def remove_fractional_faces(mesh, fractions):
    """Remove nested central holes without rounding a partial face up to deletion.

    Return exact source-face and barycentric corner correspondence for retained
    triangles. UVs are interpolated within each original face, preserving seams.
    This is intervention geometry, not a visible-defect or quality label.
    """
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (len(mesh.faces),) or not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1)):
        raise ValueError('Expected one finite removal fraction in [0,1] per face')
    source, corners = [], []
    outer = np.eye(3)
    for face, fraction in enumerate(fractions):
        if fraction == 1:
            continue
        if fraction == 0:
            source.append(face); corners.append(outer)
            continue
        inner = (1-np.sqrt(fraction))/3 + np.sqrt(fraction)*outer
        # Three trapezoids cover the ring; each has two consistently wound faces.
        for i in range(3):
            j = (i+1) % 3
            source.extend([face, face])
            corners.extend([np.stack([outer[i],outer[j],inner[j]]),
                            np.stack([outer[i],inner[j],inner[i]])])
    source = np.asarray(source, dtype=np.int64)
    corners = np.asarray(corners, dtype=np.float64).reshape(-1,3,3)
    def interpolate(values):
        return np.einsum('fij,fjk->fik', corners, values[mesh.faces[source]]).reshape(-1,values.shape[1])
    vertices = interpolate(mesh.vertices)
    normals = interpolate(mesh.normals)
    normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    result = replace(mesh, vertices=vertices, faces=np.arange(len(vertices)).reshape(-1,3),
                     normals=normals, face_materials=mesh.face_materials[source],
                     texcoords=None if mesh.texcoords is None else interpolate(mesh.texcoords))
    return result, source, corners


def deform(mesh, face_mask, amplitude, seed, smooth_steps=0, max_displacement=None):
    points, inverse, faces = topology(mesh)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    graph = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                              shape=(len(points), len(points))).tocsr()
    graph.data[:] = 1
    graph = sparse.diags(1/np.maximum(np.asarray(graph.sum(axis=1)).ravel(), 1)) @ graph
    selected = np.zeros(len(points), bool)
    selected[np.unique(faces[face_mask])] = True
    original = points.copy()
    if smooth_steps:
        weights = selected.astype(float)
        if max_displacement is not None:
            # Fade the selection boundary instead of pulling a hard cut through a wall.
            for _ in range(3):
                weights = selected * (graph @ weights)
        for _ in range(smooth_steps):
            delta = graph @ points - points
            points[selected] += amplitude*delta[selected]*weights[selected, None]
            if max_displacement is not None:
                offset = points-original
                length = np.linalg.norm(offset, axis=1)
                points = original + offset*np.minimum(1., max_displacement/np.maximum(length,1e-12))[:,None]
    else:
        field = np.random.default_rng(seed).normal(size=len(points))
        for _ in range(3):
            field = .5*field+.5*(graph @ field)
        field /= max(np.std(field), 1e-12)
        weights = selected.astype(float)
        for _ in range(3):
            weights = selected*(graph @ weights)
        normal = recompute_vertex_normals(original, faces)
        points += normal*(field*weights*amplitude)[:, None]
    # Update normals on the shared topology BEFORE restoring UV-split vertices.
    normals = recompute_vertex_normals(points, faces)[inverse]
    return replace(mesh, vertices=points[inverse], normals=normals)


def textured_qem(mesh, retained, calibrated=False):
    """MeshLab wedge-UV QEM per material, preserving material boundaries."""
    import pymeshlab as pm
    from tempfile import TemporaryDirectory
    from pathlib import Path
    origin = mesh.vertices.mean(axis=0)
    vertices, normals, uvs, materials, faces_out = [], [], [], [], []
    count = 0
    for material in np.unique(mesh.face_materials):
        f = mesh.faces[mesh.face_materials == material]
        points, inverse = np.unique(np.round(mesh.vertices[f].reshape(-1, 3)-origin, 8), axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3).astype(np.int32)
        ms = pm.MeshSet()
        # OBJ importer sets wedge texture indices; Mesh(w_tex_coords_matrix=...)
        # alone leaves them unset in MeshLab 2025.7.
        with TemporaryDirectory(prefix='v3-qem-') as temporary:
            directory = Path(temporary)
            uv = mesh.texcoords[f].reshape(-1, 2)
            paths=mesh.metadata.get('material_texture_paths')
            untextured=paths is not None and not paths[material]
            if untextured:uv=np.zeros_like(uv)
            elif not np.isfinite(uv).all():raise ValueError('Textured QEM input has invalid UV')
            lines = ['mtllib material.mtl', 'usemtl surface']
            lines += ['v '+' '.join(map(str, row)) for row in points]
            lines += ['vt '+' '.join(map(str, row)) for row in uv]
            lines += ['f '+' '.join(f'{int(v)+1}/{3*i+j+1}' for j,v in enumerate(face)) for i,face in enumerate(faces)]
            (directory/'mesh.obj').write_text('\n'.join(lines))
            (directory/'material.mtl').write_text('newmtl surface\nmap_Kd texture.png\n')
            Image.new('RGB',(2,2),'white').save(directory/'texture.png')
            ms.load_new_mesh(str(directory/'mesh.obj'))
        options=dict(targetfacenum=max(4, int(len(f)*retained)),
                     preserveboundary=True,preservenormal=True,optimalplacement=False)
        if calibrated:
            # Normal preservation can stop textured QEM long before its target.
            # Material-wise boundaries may simplify too; seam cracks must be
            # visually reviewed, never claim this is seam-preserving compression.
            options.update(preserveboundary=False, preservenormal=False, optimalplacement=True)
        if untextured:
            ms.apply_filter('meshing_decimation_quadric_edge_collapse',**options)
        else:
            ms.apply_filter('meshing_decimation_quadric_edge_collapse_with_texture',extratcoordw=1.,**options)
        result = ms.current_mesh()
        rf = result.face_matrix()
        vertices.append(result.vertex_matrix()[rf].reshape(-1, 3)+origin)
        # Keep MeshLab interpolated/shared vertex normals, never recompute per corner.
        normals.append(result.vertex_normal_matrix()[rf].reshape(-1, 3))
        uvs.append(result.wedge_tex_coord_matrix())
        faces_out.append(np.arange(count, count+3*len(rf)).reshape(-1, 3))
        count += 3*len(rf)
        materials.append(np.full(len(rf), material))
    return replace(mesh, vertices=np.vstack(vertices), faces=np.vstack(faces_out),
                   normals=np.vstack(normals), texcoords=np.vstack(uvs), face_materials=np.concatenate(materials))


def draco_positions(mesh, bits):
    import DracoPy
    origin = mesh.vertices.min(axis=0)
    stream = DracoPy.encode(np.ascontiguousarray(mesh.vertices-origin),
                            np.asarray(mesh.faces, np.uint32), quantization_bits=bits,
                            preserve_order=True, compression_level=7)
    decoded = DracoPy.decode(stream)
    if not np.array_equal(decoded.faces, mesh.faces) or len(decoded.points) != len(mesh.vertices):
        raise ValueError('Draco did not preserve attribute indexing')
    # Normal/UV attributes are deliberately retained to isolate position coding.
    return replace(mesh, vertices=np.asarray(decoded.points, float)+origin), stream


def surface_mask(mesh, selected, material, size):
    canvas = Image.new('L', size, 0)
    draw = ImageDraw.Draw(canvas)
    for face in np.flatnonzero((selected > 0) & (mesh.face_materials == material)):
        uv = mesh.texcoords[mesh.faces[face]]
        weight = float(selected[face])
        if weight < 1:
            # Homothetic triangle clipping: exact fractional surface area,
            # nested levels without changing the source mesh/UV topology.
            uv = uv[0] + (uv-uv[0])*np.sqrt(weight)
        if np.any(uv < 0) or np.any(uv > 1):
            profiles=mesh.metadata.get('material_profiles', [])
            sampler=(profiles[material].get('baseColorSampler') or {}) if material<len(profiles) else {}
            if sampler.get('wrapS',10497)!=10497 or sampler.get('wrapT',10497)!=10497:
                raise ValueError('Out-of-range UV currently supports REPEAT samplers only')
            # Rasterize translated triangle copies, not modulo each vertex:
            # vertex-wise modulo incorrectly bridges seams and changes coverage.
            lo=np.ceil(-uv.max(axis=0)).astype(int)
            hi=np.floor(1-uv.min(axis=0)).astype(int)
            if np.prod(hi-lo+1)>64:raise ValueError('UV tiling exceeds bounded pilot support')
            shifts=[(u,v) for u in range(lo[0],hi[0]+1) for v in range(lo[1],hi[1]+1)]
        else:shifts=[(0,0)]
        for shift in shifts:
            draw.polygon([(float(u*(size[0]-1)), float(v*(size[1]-1))) for u, v in uv+shift], fill=255)
    return np.asarray(canvas)>0


def island_projection_ghost(image, destination, valid_domain, relative_shift, *,
                            vertical_scale=1., shear=0., blend=.5, linear_blend=False):
    """Misprojection within raster-connected UV domains, not atlas-wide wrapping.

    Destination coverage does not shrink with displacement. Invalid source samples
    project to the nearest valid pixel in the SAME component. This is an explicit
    boundary-extension approximation, not a claim to know original camera poses.
    Raster connectivity can merge touching UV islands; record that limitation.
    Optional inverse affine sampling models local projection/parameterization
    stretch and shear. Defaults retain the previous translation-only recipe.
    """
    if not np.isfinite(relative_shift) or not 0 <= relative_shift <= 1:
        raise ValueError('relative_shift must be finite and in [0,1]')
    if not np.isfinite(vertical_scale) or not .1 <= vertical_scale <= 1:
        raise ValueError('vertical_scale must be finite and in [.1,1]')
    if not np.isfinite(shear) or abs(shear) > .5:
        raise ValueError('shear must be finite and bounded by .5')
    if not np.isfinite(blend) or not 0 <= blend <= 1:
        raise ValueError('blend must be finite and in [0,1]')
    original = np.asarray(image.convert('RGBA'))
    destination = np.asarray(destination, dtype=bool)
    valid_domain = np.asarray(valid_domain, dtype=bool)
    if destination.shape != original.shape[:2] or valid_domain.shape != destination.shape:
        raise ValueError('Texture masks must match image dimensions')
    valid_domain = valid_domain & (original[..., 3] > 0)
    components, _ = label(valid_domain)
    output = original.copy()
    for identity, box in enumerate(find_objects(components), start=1):
        if box is None:
            continue
        domain = components[box] == identity
        active = destination[box] & domain
        if not active.any():
            continue
        height, width = domain.shape
        shift = relative_shift * max(0, min(height, width)-1)
        y, x = np.nonzero(active)
        centered_y = y - (height-1)/2
        sx = np.clip(np.rint(x-shift + shear*centered_y/max(height-1,1)*(width-1)).astype(int), 0, width-1)
        sy = np.clip(np.rint(centered_y*vertical_scale + (height-1)/2-.35*shift).astype(int), 0, height-1)
        nearest = distance_transform_edt(~domain, return_distances=False, return_indices=True)
        source_y, source_x = nearest[:, sy, sx]
        source = original[box]
        original_rgb = source[y, x, :3].astype(float)/255
        shifted_rgb = source[source_y, source_x, :3].astype(float)/255
        if linear_blend:
            original_rgb = np.where(original_rgb <= .04045, original_rgb/12.92, ((original_rgb+.055)/1.055)**2.4)
            shifted_rgb = np.where(shifted_rgb <= .04045, shifted_rgb/12.92, ((shifted_rgb+.055)/1.055)**2.4)
        mixed = (1-blend)*original_rgb + blend*shifted_rgb
        if linear_blend:
            mixed = np.where(mixed <= .0031308, mixed*12.92, 1.055*mixed**(1/2.4)-.055)
        output[box][y, x, :3] = np.rint(np.clip(mixed,0,1)*255).astype(np.uint8)
    return Image.fromarray(output)


def exposure_inconsistency(image, mask, exposure_ev, warmth=0.):
    """Bounded linear-light exposure/white-balance mismatch, preserving alpha."""
    if not np.isfinite(exposure_ev) or abs(exposure_ev) > 4:
        raise ValueError('Exposure must be finite and within four stops')
    if not np.isfinite(warmth) or abs(warmth) > .2:
        raise ValueError('White-balance shift must be finite and bounded')
    array = np.asarray(image.convert('RGBA')).copy()
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != array.shape[:2]:
        raise ValueError('Texture mask must match image dimensions')
    rgb = array[..., :3].astype(float)/255
    linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
    linear *= 2.**exposure_ev * np.array([1+warmth, 1., 1-warmth])
    linear = np.clip(linear, 0, 1)
    rgb = np.where(linear <= .0031308, linear*12.92, 1.055*linear**(1/2.4)-.055)
    array[mask, :3] = np.rint(np.clip(rgb[mask]*255, 0, 255)).astype(np.uint8)
    return Image.fromarray(array)


def local_texture(image, mask, kind, strength):
    array = np.asarray(image.convert('RGBA')).copy()
    if kind == 'missing':
        array[mask, :3] = 128
    elif kind == 'seam':
        rgb = array[..., :3].astype(float)/255
        linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        linear *= np.array([1+strength, 1+strength*.5, 1-strength*.25])
        rgb = np.where(linear <= .0031308, linear*12.92, 1.055*np.maximum(linear, 0)**(1/2.4)-.055)
        array[mask, :3] = np.clip(rgb[mask]*255, 0, 255).astype(np.uint8)
    elif kind == 'misalignment':
        y, x = np.indices(mask.shape, dtype=float)
        sx, sy = x-strength, y-strength*.35
        valid = map_coordinates(mask.astype(float), [sy, sx], order=0, mode='constant', cval=0)>0
        active = mask & valid
        for channel in range(3):
            shifted = map_coordinates(array[..., channel].astype(float), [sy, sx], order=1, mode='nearest')
            array[..., channel][active] = np.rint(.5*array[..., channel][active]+.5*shifted[active]).astype(np.uint8)
    else:
        raise ValueError(kind)
    return Image.fromarray(array)
