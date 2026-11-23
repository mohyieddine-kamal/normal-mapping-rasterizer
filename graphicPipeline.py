import numpy as np

# ---------------------------------------------------------------------------
# Utility: sample a texture at UV coordinates (u, v)
# texture.shape is (height, width, channels): u spans the width, v the height
# ---------------------------------------------------------------------------
def sample(texture, u, v):
    height, width = texture.shape[0], texture.shape[1]
    x = int(u * width)
    y = int((1 - v) * height)
    # Clamp to valid pixel range to avoid index-out-of-bounds at edges
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return texture[y, x] / 255.0


# ---------------------------------------------------------------------------
# Fragment: one pixel candidate produced by the Rasterizer
# ---------------------------------------------------------------------------
class Fragment:
    def __init__(self, x: int, y: int, depth: float, interpolated_data):
        self.x = x
        self.y = y
        self.depth = depth
        self.interpolated_data = interpolated_data
        self.output = []


# ---------------------------------------------------------------------------
# Edge-side function: used for barycentric / inside-triangle test
# ---------------------------------------------------------------------------
def edgeSide(p, v0, v1):
    return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])


# ---------------------------------------------------------------------------
# GraphicPipeline
# ---------------------------------------------------------------------------
class GraphicPipeline:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.image = np.zeros((height, width, 3))
        self.depthBuffer = np.ones((height, width))

    # -----------------------------------------------------------------------
    # ComputeTangent
    # Input vertex layout  (8 values):  [0:3] pos  [3:6] normal  [6:8] UV
    # Output vertex layout (11 values): same + [8:11] tangent T
    # -----------------------------------------------------------------------
    def ComputeTangent(self, vertices, triangles):
        tangents   = np.zeros((vertices.shape[0], 3))

        for tr in triangles:
            v0, v1, v2 = vertices[tr[0]], vertices[tr[1]], vertices[tr[2]]

            edge1    = v1[0:3] - v0[0:3]
            edge2    = v2[0:3] - v0[0:3]
            uv_edge1 = v1[6:8] - v0[6:8]
            uv_edge2 = v2[6:8] - v0[6:8]

            uv_area = uv_edge1[0] * uv_edge2[1] - uv_edge1[1] * uv_edge2[0]

            # Guard against degenerate UV triangles
            if abs(uv_area) < 1e-8:
                continue

            tangent   = (uv_edge2[1] * edge1 - uv_edge1[1] * edge2) / uv_area

            for v in tr:
                tangents[v]   += tangent

        for i in range(vertices.shape[0]):
            n = vertices[i, 3:6]
            t = tangents[i]

            norm_t = np.linalg.norm(t)
            if norm_t < 1e-8:
                arbitrary = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(n, arbitrary)) > 0.9:
                    arbitrary = np.array([0.0, 1.0, 0.0])
                t = np.cross(n, arbitrary)

            t = t - np.dot(n, t) * n
            norm_t = np.linalg.norm(t)
            t = t / norm_t if norm_t >= 1e-8 else np.array([1.0, 0.0, 0.0])

            tangents[i]   = t

        vertices_w_tang = np.zeros((vertices.shape[0], 11))
        vertices_w_tang[:, 0:8]   = vertices[:, 0:8]
        vertices_w_tang[:, 8:11]  = tangents
        return vertices_w_tang

    # -----------------------------------------------------------------------
    # VertexShader
    # Input  (11 values): pos[0:3] normal[3:6] UV[6:8] T[8:11]
    # Output (17 values):
    #   [0:3]   NDC position
    #   [3:6]   normal
    #   [6:9]   view direction  V = camPos - vertPos
    #   [9:12]  light direction L = lightPos - vertPos
    #   [12:14] UV
    #   [14:17] tangent T
    # -----------------------------------------------------------------------
    def VertexShader(self, vertex, data):
        out = np.zeros(17)

        pos_clip = data['projMatrix'] @ (data['viewMatrix'] @ np.array([*vertex[0:3], 1.0]))
        out[0] = pos_clip[0] / pos_clip[3]
        out[1] = pos_clip[1] / pos_clip[3]
        out[2] = pos_clip[2] / pos_clip[3]

        out[3:6]  = vertex[3:6]                          # normal
        out[6:9]  = data['cameraPosition'] - vertex[0:3] # view dir
        out[9:12] = data['lightPosition']  - vertex[0:3] # light dir
        out[12:14] = vertex[6:8]                         # UV
        out[14:17] = vertex[8:11]                        # tangent T

        return out

    # -----------------------------------------------------------------------
    # Rasterizer
    # -----------------------------------------------------------------------
    def Rasterizer(self, v0, v1, v2):
        fragments = []

        area = edgeSide(v0, v1, v2)
        if area <= 0:
            return fragments

        def to_px(v):
            return np.array([
                (v[0] + 1.0) / 2.0 * self.width,
                (v[1] + 1.0) / 2.0 * self.height
            ])

        v0_px, v1_px, v2_px = to_px(v0), to_px(v1), to_px(v2)

        A = np.floor(np.min([v0_px, v1_px, v2_px], axis=0)).astype(int)
        B = np.ceil( np.max([v0_px, v1_px, v2_px], axis=0)).astype(int)
        A = np.maximum(A, [0, 0])
        B = np.minimum(B, [self.width - 1, self.height - 1])

        l = v0.shape[0]

        for j in range(A[1], B[1] + 1):
            for i in range(A[0], B[0] + 1):
                x = (i + 0.5) / self.width  * 2.0 - 1.0
                y = (j + 0.5) / self.height * 2.0 - 1.0
                p = np.array([x, y])

                a0 = edgeSide(p, v0, v1)
                a1 = edgeSide(p, v1, v2)
                a2 = edgeSide(p, v2, v0)

                if a0 >= 0 and a1 >= 0 and a2 >= 0:
                    lam0 = a1 / area
                    lam1 = a2 / area
                    lam2 = a0 / area

                    z = lam0 * v0[2] + lam1 * v1[2] + lam2 * v2[2]
                    interp = v0[3:l] * lam0 + v1[3:l] * lam1 + v2[3:l] * lam2

                    fragments.append(Fragment(i, j, z, interp))

        return fragments

    # -----------------------------------------------------------------------
    # fragmentShader
    #
    # interpolated_data layout (= outputVertex[3:], 14 values):
    #   [0:3]   normal N
    #   [3:6]   view direction V
    #   [6:9]   light direction L
    #   [9]     u
    #   [10]    v
    #   [11:14] tangent T
    # -----------------------------------------------------------------------
    def fragmentShader(self, fragment, data):
        d = fragment.interpolated_data

        N = d[0:3];  N = N / (np.linalg.norm(N) + 1e-8)
        V = d[3:6];  V = V / (np.linalg.norm(V) + 1e-8)
        L = d[6:9];  L = L / (np.linalg.norm(L) + 1e-8)
        u, v = d[9], d[10]


        T = d[11:14]
        # Re-orthogonalise T against N after interpolation
        T = T - np.dot(T, N) * N
        norm_T = np.linalg.norm(T)
        if norm_T < 1e-8:
            arb = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(N, arb)) > 0.9:
                arb = np.array([0.0, 1.0, 0.0])
            T = np.cross(N, arb)
            T /= np.linalg.norm(T)
        else:
            T /= norm_T

        B = np.cross(N, T)  # recompute for guaranteed orthonormality

        # TBN: columns are T, B, N  →  maps tangent space → world space
        TBN = np.column_stack([T, B, N])

        # Sample normal map and decode: [0,1] → [-1,1]
        nm_raw = sample(data['normalMap'], u, v)[:3]
        nm_tangent = nm_raw * 2.0 - 1.0
        nm_tangent = nm_tangent / (np.linalg.norm(nm_tangent) + 1e-8)

        if not data.get('useNormalMap', True):
            nm_tangent = np.array([0.0, 0.0, 1.0])         # flat → no normal mapping

        N_final = TBN @ nm_tangent
        N_final = N_final / (np.linalg.norm(N_final) + 1e-8)

        # Phong lighting
        diffuse  = max(np.dot(N_final, L), 0.0)
        R        = 2.0 * np.dot(L, N_final) * N_final - L
        specular = pow(max(np.dot(R, V), 0.0), 64)


        ka, kd, ks = 0.05, 0.85, 0.30
        phong = ka + kd * diffuse + ks * specular

        # Toon / cel shading quantisation (kept from original design)
        phong = float(np.clip(np.ceil(phong * 4.0 + 1.0) / 6.0, 0.0, 1.0))

        texture = sample(data['texture'], u, v)[:3]
        fragment.output = np.array([phong, phong, phong]) * texture

    # -----------------------------------------------------------------------
    # draw — main entry point
    # -----------------------------------------------------------------------
    def draw(self, vertices, triangles, data):
        vertices = self.ComputeTangent(vertices, triangles)

        self.newVertices = np.zeros((vertices.shape[0], 17))
        for i in range(vertices.shape[0]):
            self.newVertices[i] = self.VertexShader(vertices[i], data)

        fragments = []
        for tri in triangles:
            fragments.extend(self.Rasterizer(
                self.newVertices[tri[0]],
                self.newVertices[tri[1]],
                self.newVertices[tri[2]]
            ))

        # Early depth test: only shade fragments that are closer than what is
        # already stored. Same final image, but hidden fragments are never shaded.
        for f in fragments:
            if self.depthBuffer[f.y][f.x] > f.depth:
                self.fragmentShader(f, data)
                self.depthBuffer[f.y][f.x] = f.depth
                self.image[f.y][f.x] = f.output
