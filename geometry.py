import numpy as np
from scipy.spatial import ConvexHull
import ot  # Make sure POT is installed: pip install POT

def compute_square_geometry(square):
    """Compute geometric properties including convex hull of a magic square in 3D space"""
    n = square.shape[0]
    # Create 3D points (x,y,value)
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    points = np.array([(x[i, j], y[i, j], square[i, j]) 
                       for i in range(n) for j in range(n)])
    
    try:
        # Compute convex hull
        hull = ConvexHull(points)
        
        # Store hull vertices and simplices directly
        hull_vertices = points[hull.vertices]
        hull_faces = [points[simplex] for simplex in hull.simplices]
        
        # Compute centroid and relative positions
        centroid = np.mean(points, axis=0)
        rel_positions = points - centroid
        
        # Compute covariance matrix and eigendecomposition
        cov_matrix = np.cov(rel_positions.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by absolute eigenvalue
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        
        # Ensure consistent orientation
        for i in range(3):
            if eigenvectors[0,i] < 0:
                eigenvectors[:,i] *= -1
        
        # Compute distances from centroid
        distances = np.linalg.norm(rel_positions, axis=1)
        
        # Compute angles (avoiding potential NaN)
        angles = np.arctan2(rel_positions[:,1], rel_positions[:,0])
        angles = angles[~np.isnan(angles)]  # Remove any NaN values
        
        # Ensure we don't divide by zero
        hull_volume = max(hull.volume, 1e-10)
        hull_area = max(hull.area, 1e-10)
        
        shape_descriptors = {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'max_distance': np.max(distances),
            'min_distance': np.min(distances),
            'angular_uniformity': np.std(angles) if len(angles) > 0 else 0,
            'volume_area_ratio': hull_area / hull_volume,
            'compactness': (hull_area ** 3) / (36 * np.pi * hull_volume ** 2)
        }
        
        return {
            'points': points,
            'hull_vertices': hull_vertices,
            'hull_faces': hull_faces,
            'centroid': centroid,
            'cov_matrix': cov_matrix,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'hull_volume': hull_volume,
            'hull_area': hull_area,
            'shape_descriptors': shape_descriptors
        }
        
    except Exception as e:
        print(f"Error computing hull: {e}")
        print(f"Square:\n{square}")
        return None

def point_on_face(point, face_points, tolerance=1e-10):
    """Check if a point lies on a triangular face"""
    # Calculate normal vector of the face
    v1 = face_points[1] - face_points[0]
    v2 = face_points[2] - face_points[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    # Check if point lies on the plane of the face
    v = point - face_points[0]
    dist_to_plane = abs(np.dot(v, normal))
    if dist_to_plane > tolerance:
        return False
    
    # Check if point is within the triangle
    # Using barycentric coordinates
    def same_side(p1, p2, a, b):
        cp1 = np.cross(b-a, p1-a)
        cp2 = np.cross(b-a, p2-a)
        return np.dot(cp1, cp2) >= 0
    
    return (same_side(point, face_points[0], face_points[1], face_points[2]) and
            same_side(point, face_points[1], face_points[0], face_points[2]) and
            same_side(point, face_points[2], face_points[0], face_points[1]))

def categorize_points(geo):
    """Categorize points as vertices, face points, or interior points"""
    points = geo['points']
    hull_vertices = geo['hull_vertices']
    faces = geo['hull_faces']
    
    vertex_points = []
    face_points = []
    interior_points = []
    
    for point in points:
        if any(np.allclose(point, vertex) for vertex in hull_vertices):
            vertex_points.append(point)
        else:
            # Check if point lies on any face
            on_face = False
            for face in faces:
                if point_on_face(point, face, tolerance=1e-10):
                    on_face = True
                    break
            if on_face:
                face_points.append(point)
            else:
                interior_points.append(point)
    
    return {
        'vertex_points': np.array(vertex_points) if vertex_points else np.empty((0, 3)),
        'face_points': np.array(face_points) if face_points else np.empty((0, 3)),
        'interior_points': np.array(interior_points) if interior_points else np.empty((0, 3))
    }

def analyze_patterns(squares):
    """Analyze point distribution patterns across squares"""
    patterns = {}
    total = len(squares)
    pattern_examples = {}  # Store example squares for each pattern
    
    # Collect counts for statistics
    vertex_counts = []
    face_counts = []
    interior_counts = []
    
    print(f"\nAnalyzing {total} squares...")
    for i, square in enumerate(squares):
        if i % 100 == 0:
            print(f"Processing square {i}...")
            
        geo = compute_square_geometry(square)
        if geo is None:
            continue
            
        cats = categorize_points(geo)
        
        n_vertices = len(cats['vertex_points'])
        n_faces = len(cats['face_points'])
        n_interior = len(cats['interior_points'])
        
        vertex_counts.append(n_vertices)
        face_counts.append(n_faces)
        interior_counts.append(n_interior)
        
        key = (n_vertices, n_faces, n_interior)
        patterns[key] = patterns.get(key, 0) + 1
        
        # Store the first example of each pattern
        if key not in pattern_examples:
            pattern_examples[key] = square
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Vertices:    {np.mean(vertex_counts):.1f} ± {np.std(vertex_counts):.1f}")
    print(f"Face points: {np.mean(face_counts):.1f} ± {np.std(face_counts):.1f}")
    print(f"Interior:    {np.mean(interior_counts):.1f} ± {np.std(interior_counts):.1f}")
    
    # Print pattern distribution
    print("\nDistribution Patterns:")
    print("Vertices | Face Points | Interior | Count | Percentage")
    print("-" * 60)
    for (v, f, i), count in sorted(patterns.items()):
        percentage = (count / total) * 100
        print(f"{v:8d} | {f:11d} | {i:8d} | {count:5d} | {percentage:6.1f}%")
    
    # Print example squares for each pattern
    print("\nExample Squares for Each Pattern:")
    print("-" * 60)
    for (v, f, i), square in sorted(pattern_examples.items()):
        print(f"\nPattern: {v} vertices, {f} face points, {i} interior points")
        print(square)
    
    return patterns, {
        'vertex_counts': vertex_counts,
        'face_counts': face_counts,
        'interior_counts': interior_counts,
        'pattern_examples': pattern_examples  # Include examples in return value
    }