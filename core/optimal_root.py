from collections import deque

def _bfs(start, adj, n):
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    far = start
    while q:
        u = q.popleft()
        for nbr, *_ in adj[u]:       # ← unpack (nbr, la, lb), ignore la/lb
            if dist[nbr] == -1:
                dist[nbr] = dist[u] + 1
                if dist[nbr] > dist[far]:
                    far = nbr
                q.append(nbr)
    return dist, far

def optimal_root(dual_adj, n_tris):
    best_root, best_depth = 0, float('inf')
    for f in range(n_tris):
        dist, _ = _bfs(f, dual_adj, n_tris)
        d = max(dist)
        if d < best_depth:
            best_depth, best_root = d, f
    return best_root, best_depth

def optimal_root_fast(dual_adj, n_tris):
    """Double-BFS: returns (centre, eccentricity). Centre minimises max depth."""
    _,      u = _bfs(0, dual_adj, n_tris)
    dist_u, v = _bfs(u, dual_adj, n_tris)
    dist_v, _ = _bfs(v, dual_adj, n_tris)
    best = min(range(n_tris), key=lambda i: max(dist_u[i], dist_v[i]))
    return best, max(dist_u[best], dist_v[best])

def optimal_root_endpoint(dual_adj, n_tris):
    """
    Double-BFS: returns one diameter endpoint.

    For elongated meshes (nose cones, cylinders), rooting at a diameter
    endpoint distributes BFS depth more evenly across circumferential
    stitch pairs than rooting at the centre, reducing 3D reconstruction
    error accumulation.
    """
    _,      u = _bfs(0, dual_adj, n_tris)
    dist_u, v = _bfs(u, dual_adj, n_tris)
    # v is the farthest point from u — a diameter endpoint.
    # Use v as root (the base-rim triangle for a nose cone).
    ecc = max(dist_u[v], 0)
    return v, ecc