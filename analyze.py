import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from core import (
    load_latest_squares,
    reduce_to_unique,
    validate_normalization
)
from geometry import (
    compute_square_geometry,
    categorize_points,
    analyze_patterns
)
from visualization import (
    plot_single_hull,
    plot_multiple_squares,
    plot_eigenvectors,
    plot_comprehensive_analysis,
    plot_tsne_emd,
    plot_geometric_tsne_with_emd,
    analyze_distance_distributions,
    estimate_manifold_dimension,
    analyze_manifold
)

def plot_tsne_analysis(squares, feature_type='raw'):
    """Compute and plot t-SNE analysis of squares
    
    Args:
        squares: List of magic squares
        feature_type: 'raw' or 'geometric'
    """
    print(f"\nComputing t-SNE using {feature_type} features...")
    
    if feature_type == 'geometric':
        # Compute geometric features
        features = []
        geometries = []
        for i, square in enumerate(squares):
            if i % 100 == 0:
                print(f"Processing square {i+1}/{len(squares)}...")
            geo = compute_square_geometry(square)
            cats = categorize_points(geo)
            n_vertices = len(cats['vertex_points'])
            n_faces = len(cats['face_points'])
            n_interior = len(cats['interior_points'])
            geometries.append((n_vertices, n_faces, n_interior))
            
            features.append([
                n_vertices,
                n_faces,
                n_interior,
                geo['hull_volume'],
                geo['hull_area'],
                geo['eigenvalues'][0],
                geo['eigenvalues'][1],
                geo['eigenvalues'][2]
            ])
        X = np.array(features)
    else:
        # Use raw square values
        X = np.array(squares).reshape(len(squares), -1)
        geometries = []
        for square in squares:
            geo = compute_square_geometry(square)
            cats = categorize_points(geo)
            geometries.append((
                len(cats['vertex_points']),
                len(cats['face_points']),
                len(cats['interior_points'])
            ))
    
    print(f"Input shape: {X.shape}")
    
    # Standardize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Color by geometric properties
    vertices = [g[0] for g in geometries]
    faces = [g[1] for g in geometries]
    interior = [g[2] for g in geometries]
    
    # Plot with different colorings
    sc1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=vertices, 
                     cmap='plasma', s=10, alpha=0.6)
    ax1.set_title('Colored by Number of Vertices')
    plt.colorbar(sc1, ax=ax1)
    
    sc2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=faces, 
                     cmap='plasma', s=10, alpha=0.6)
    ax2.set_title('Colored by Number of Face Points')
    plt.colorbar(sc2, ax=ax2)
    
    sc3 = ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=interior, 
                     cmap='plasma', s=10, alpha=0.6)
    ax3.set_title('Colored by Number of Interior Points')
    plt.colorbar(sc3, ax=ax3)
    
    plt.suptitle(f't-SNE Visualization Using {feature_type.title()} Features', 
                 fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return X_tsne, geometries

def main():
    # Load squares
    squares = load_latest_squares()
    if squares is not None:
        # Reduce to 880 unique squares
        print("\n=== Reducing to 880 Unique Squares ===")
        reduced_squares = reduce_to_unique(squares)
        
        # Normalize the 880 squares
        print("\n=== Normalizing 880 Squares ===")
        normalized_squares = validate_normalization(reduced_squares)
                # Add pattern analysis
        print("\n=== Analyzing Square Patterns ===")
        patterns_results = analyze_patterns(normalized_squares)
        
        # Compute t-SNE with raw features
        print("\n=== Computing t-SNE (Raw Features) ===")
        tsne_raw, geom_raw = plot_tsne_analysis(normalized_squares, 'raw')
        
        # Compute t-SNE with geometric features
        print("\n=== Computing t-SNE (Geometric Features) ===")
        tsne_geo, geom_geo = plot_tsne_analysis(normalized_squares, 'geometric')
        
        # Plot eigenvectors
        print("\n=== Computing Eigenvector Analysis ===")
        eigendata = plot_eigenvectors(normalized_squares)
        
        # Add comprehensive analysis
        print("\n=== Computing Comprehensive Analysis ===")
        analysis_df = plot_comprehensive_analysis(normalized_squares)
        
        # Add EMD-based t-SNE analysis
        print("\n=== Computing EMD-based t-SNE Analysis ===")
        distances, embedding, threshold = plot_tsne_emd(
            normalized_squares,
            perplexity=30,
            distance_threshold=None  # Will use 1th percentile by default
        )
        
        # Add distance distribution analysis with caching
        print("\n=== Analyzing Magic Square vs Random Distance Distributions ===")
        magic_distances, random_distances, emd_matrix = analyze_distance_distributions(
            normalized_squares,
            n_random=100,
            cache_file='emd_distances.pkl'
        )
        
        # Add manifold analysis
        print("\n=== Analyzing Magic Square Manifold ===")
        manifold_results = analyze_manifold(normalized_squares)
        
    # Move the return statement to the end of the function
    return (normalized_squares, embedding, distances, threshold, 
            magic_distances, random_distances, manifold_results, patterns_results)

if __name__ == "__main__":
    (squares, embedding, distances, threshold, magic_distances, 
     random_distances, manifold_results, patterns_results) = main()
    print("\nAnalysis complete!")
    print("Variables available:")
    print("- squares: normalized magic squares")
    print("- distances: EMD distances between squares")
    print("- embedding: t-SNE embedding using EMD")
    print("- threshold: distance threshold used for EMD-based t-SNE")
    print("- magic_distances: EMD distances between magic squares")
    print("- random_distances: EMD distances between magic squares and random permutations")
    print("- emd_matrix: EMD distance matrix between all magic squares")
    print("- manifold_results: results of manifold analysis")
    print("- patterns_results: results of pattern analysis")