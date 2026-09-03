# Pseudo-MOS v3 Pilot

| 攻击 | 等级 | OQI v2 | 感知质量 | v3几何 | v3纹理 | v3综合 |
|---|---|---:|---:|---:|---:|---:|
| geometry_hole | light | 0.617 | 0.575 | 0.616 | 1.000 | 0.657 |
| geometry_hole | medium | 0.130 | 0.323 | 0.326 | 1.000 | 0.368 |
| geometry_hole | heavy | 0.019 | 0.236 | 0.217 | 1.000 | 0.259 |
| geometry_noise_spike | light | 0.636 | 0.717 | 0.702 | 1.000 | 0.741 |
| geometry_noise_spike | medium | 0.461 | 0.555 | 0.559 | 1.000 | 0.600 |
| geometry_noise_spike | heavy | 0.343 | 0.395 | 0.433 | 1.000 | 0.474 |
| mesh_simplification_qem | light | 0.896 | 0.879 | 0.889 | 1.000 | 0.914 |
| mesh_simplification_qem | medium | 0.758 | 0.570 | 0.642 | 1.000 | 0.682 |
| mesh_simplification_qem | heavy | 0.569 | 0.410 | 0.485 | 1.000 | 0.526 |
| texture_detail_loss | light | 0.882 | 0.919 | 1.000 | 0.903 | 0.925 |
| texture_detail_loss | medium | 0.740 | 0.716 | 1.000 | 0.740 | 0.777 |
| texture_detail_loss | heavy | 0.633 | 0.557 | 1.000 | 0.608 | 0.649 |
| texture_misalignment | light | 0.943 | 0.980 | 1.000 | 0.961 | 0.972 |
| texture_misalignment | medium | 0.845 | 0.868 | 1.000 | 0.862 | 0.890 |
| texture_misalignment | heavy | 0.716 | 0.693 | 1.000 | 0.718 | 0.757 |
| texture_region_missing | light | 0.607 | 0.639 | 1.000 | 0.652 | 0.692 |
| texture_region_missing | medium | 0.378 | 0.310 | 1.000 | 0.378 | 0.420 |
| texture_region_missing | heavy | 0.227 | 0.206 | 1.000 | 0.274 | 0.316 |

三级单调：6/6类通过。

该结果仅用于验证公式形态；正式尺度必须在扩容后的Train上冻结。
