"""
SVG Exporter for NetFold
========================
Generates a machine-readable laser-cutting SVG pattern from a 2D NetFold.
Conventions:
  - Red (#FF0000), 0.1px: Outer boundaries (Cut operations)
  - Blue (#0000FF), 0.1px: Inner folds (Score/Vector Engrave operations)
  - Black (#000000), fill: Text labels (Raster Engrave operations)
"""

import numpy as np
import math
from .structure import NetFold

def export_svg(nf: NetFold, save_path: str, scale_multiplier: float = 100.0, padding: float = 20.0):
    """
    Exports a NetFold object to an SVG file. If the NetFold has multiple disconnected
    islands, it will export each island to a separate SVG file.
    
    Args:
        nf: The NetFold object.
        save_path: File path to save the SVG to. For multiple islands, '_island_X' is appended.
        scale_multiplier: Scales the [0, 1] normalized coordinates to SVG units (e.g. 100 pixels).
        padding: Padding around the edge of the SVG canvas.
    Returns:
        List of generated file paths.
    """
    if not nf.triangles:
        raise ValueError("NetFold has no triangles to export.")

    # Determine islands
    if not nf.islands:
        islands = [[t.id for t in nf.triangles]]
    else:
        islands = nf.islands

    saved_paths = []
    
    for idx, island_tri_ids in enumerate(islands):
        island_tris = set(island_tri_ids)
        
        # Filter geometry for this specific island
        tris_in_island = [t for t in nf.triangles if t.id in island_tris]
        if not tris_in_island:
            continue
            
        tri_id_map = {t.id: t for t in tris_in_island}
            
        # 1. Gather all 2D points to compute the bounding box for THIS island
        all_points = []
        for tri in tris_in_island:
            all_points.extend(tri.vertices)
        all_points = np.array(all_points)
        
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)
        
        width_raw = max_x - min_x
        height_raw = max_y - min_y
        
        # SVG Dimensions tightly cropped to this island
        svg_width = width_raw * scale_multiplier + 2 * padding
        svg_height = height_raw * scale_multiplier + 2 * padding
        
        # Helper to map 2D NetFold coordinates to SVG coordinates
        def map_pt(pt):
            x = (pt[0] - min_x) * scale_multiplier + padding
            # Flip Y
            y = svg_height - ((pt[1] - min_y) * scale_multiplier + padding)
            return x, y

        # 2. Identify cut edges (outer boundaries) for this island
        edge_counts = {}
        for tri in tris_in_island:
            for i in range(3):
                p1 = tuple(tri.vertices[i])
                p2 = tuple(tri.vertices[(i+1)%3])
                canon = tuple(sorted([p1, p2]))
                edge_counts[canon] = edge_counts.get(canon, 0) + 1

        cut_edges = []
        for edge, count in edge_counts.items():
            if count == 1:
                cut_edges.append(edge)
                
        # Identify fold edges belonging to this island
        fold_edges_in_island = [
            fe for fe in nf.fold_edges 
            if fe.tri_a in island_tris and fe.tri_b in island_tris
        ]

        # 3. Start generating SVG XML
        svg_lines = []
        svg_lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
        svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{svg_width:.2f}" height="{svg_height:.2f}">')
        
        # White background (useful for viewing, laser cutters ignore white fill)
        svg_lines.append(f'  <rect width="100%" height="100%" fill="#FFFFFF"/>')
        
        # Group for Score lines (Fold edges) - BLUE (#0000FF), 0.1px stroke
        svg_lines.append('  <!-- SCORE LINES (Inner Folds) -->')
        svg_lines.append('  <g stroke="#0000FF" stroke-width="0.1" stroke-dasharray="2,2" fill="none">')
        
        fold_labels = []
        
        for fe in fold_edges_in_island:
            tri_a = tri_id_map[fe.tri_a]
            p1 = tri_a.vertices[fe.local_a[0]]
            p2 = tri_a.vertices[fe.local_a[1]]
            
            x1, y1 = map_pt(p1)
            x2, y2 = map_pt(p2)
            
            # Calculate physical folding angle from flat
            bend_angle_rad = math.pi - fe.dihedral_angle
            bend_angle_deg = math.degrees(bend_angle_rad) * fe.fold_direction
            
            # Skip drawing score lines if the surface is completely flat (e.g., < 0.1 degrees)
            # This perfectly fuses flat areas (like wings) into a single piece of metal!
            if abs(bend_angle_deg) > 0.1:
                svg_lines.append(f'    <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"/>')
                
                # Only print labels for bends sharper than 5 degrees
                if abs(bend_angle_deg) > 5.0:
                    # Calculate line length in SVG units to scale font
                    dx = x2 - x1
                    dy = y2 - y1
                    line_length = math.sqrt(dx**2 + dy**2)
                    
                    # Dynamic font scaling
                    font_size = min(12.0, line_length * 0.4)
                    
                    # Skip labels on extremely tiny folds where text would be an unreadable dot
                    if font_size >= 2.0:
                        mid_x = (x1 + x2) / 2
                        mid_y = (y1 + y2) / 2
                        
                        text_angle = math.degrees(math.atan2(dy, dx))
                        # Keep text readable (right-side up)
                        if text_angle > 90:
                            text_angle -= 180
                        elif text_angle < -90:
                            text_angle += 180
                            
                        fold_labels.append((mid_x, mid_y, text_angle, f"{bend_angle_deg:+.1f}°", font_size))
            
        svg_lines.append('  </g>')
        
        # Group for Cut lines (Outer boundaries) - RED (#FF0000), 0.1px stroke
        svg_lines.append('  <!-- CUT LINES (Outer Boundaries) -->')
        svg_lines.append('  <g stroke="#FF0000" stroke-width="0.1" fill="none">')
        for p1, p2 in cut_edges:
            x1, y1 = map_pt(p1)
            x2, y2 = map_pt(p2)
            svg_lines.append(f'    <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"/>')
        svg_lines.append('  </g>')
        
        # Group for Engrave lines (Text Labels) - BLACK (#000000) fill, no stroke
        svg_lines.append('  <!-- ENGRAVE LINES (Text Labels) -->')
        svg_lines.append('  <g fill="#000000" font-family="Arial" text-anchor="middle" stroke="none">')
        for mid_x, mid_y, text_angle, label, font_size in fold_labels:
            offset_y = mid_y - (font_size * 0.3)  # Hover slightly above the line
            svg_lines.append(f'    <text x="{mid_x:.2f}" y="{offset_y:.2f}" font-size="{font_size:.1f}" transform="rotate({text_angle:.1f} {mid_x:.2f},{mid_y:.2f})">{label}</text>')
        svg_lines.append('  </g>')
        
        svg_lines.append('</svg>')
        
        # Determine path for this island
        if len(islands) == 1:
            curr_path = save_path
        else:
            if save_path.lower().endswith('.svg'):
                curr_path = f"{save_path[:-4]}_island_{idx}.svg"
            else:
                curr_path = f"{save_path}_island_{idx}"
                
        with open(curr_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(svg_lines))
            
        saved_paths.append(curr_path)
    
    return saved_paths
