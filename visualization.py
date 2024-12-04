import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.manifold import TSNE
import seaborn as sns
import ot  # Make sure POT is installed: pip install POT
from scipy import stats
import os
import pickle
from sklearn.neighbors import NearestNeighbors
from scipy.stats import linregress

from geometry import compute_square_geometry, categorize_points

def plot_single_hull(square, title="Single Magic Square Hull"):
    """Plot a single magic square's points and convex hull"""
    # Compute geometry
    geo = compute_square_geometry(square)
    if geo is None:
        print("Failed to compute hull")
        return
        
    # Create 3D plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    
    # Get categorized points
    cats = categorize_points(geo)
    
    # Plot points with different colors
    if len(cats['vertex_points']) > 0:
        ax.scatter(cats['vertex_points'][:, 0], cats['vertex_points'][:, 1], cats['vertex_points'][:, 2], 
                  c='red', s=150, label=f'Hull Vertices ({len(cats["vertex_points"])})', zorder=4)
    if len(cats['face_points']) > 0:
        ax.scatter(cats['face_points'][:, 0], cats['face_points'][:, 1], cats['face_points'][:, 2], 
                  c='orange', s=100, label=f'Face Points ({len(cats["face_points"])})', zorder=3)
    if len(cats['interior_points']) > 0:
        ax.scatter(cats['interior_points'][:, 0], cats['interior_points'][:, 1], cats['interior_points'][:, 2], 
                  c='blue', s=80, label=f'Interior Points ({len(cats["interior_points"])})', zorder=2)
    
    # Plot hull faces
    faces = geo['hull_faces']
    poly3d = [[face[0], face[1], face[2]] for face in faces]
    collection = Poly3DCollection(poly3d, 
                                alpha=0.15,
                                linewidth=2,
                                edgecolor='black',
                                zorder=1)
    
    light_blue = colors.to_rgba('lightblue', alpha=0.15)
    collection.set_facecolor([light_blue])
    collection.set_zsort('average')
    ax.add_collection3d(collection)
    
    # Labels and formatting
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')
    ax.set_title(title)
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=20, azim=45)
    ax.legend()
    
    # Print statistics
    print("\nPoint Distribution:")
    print(f"Total points: {len(geo['points'])}")
    print(f"Hull vertices: {len(cats['vertex_points'])} ({len(cats['vertex_points'])/len(geo['points'])*100:.1f}%)")
    print(f"Points on faces: {len(cats['face_points'])} ({len(cats['face_points'])/len(geo['points'])*100:.1f}%)")
    print(f"Interior points: {len(cats['interior_points'])} ({len(cats['interior_points'])/len(geo['points'])*100:.1f}%)")
    print("\nMagic Square:")
    print(square)
    
    # Add interaction
    def on_move(event):
        if event.inaxes == ax:
            ax.view_init(elev=ax.elev, azim=ax.azim)
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    
    plt.show()

def plot_multiple_squares(squares, grid_size=None):
    """Plot all squares in a grid with minimal decoration"""
    # Calculate appropriate grid size if not specified
    n_squares = len(squares)
    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(n_squares)))
    
    # Verify we have enough squares
    total_slots = grid_size * grid_size
    if total_slots < n_squares:
        print(f"Warning: Grid size {grid_size}x{grid_size}={total_slots} is too small for {n_squares} squares")
        grid_size = int(np.ceil(np.sqrt(n_squares)))
        total_slots = grid_size * grid_size
        print(f"Adjusted to {grid_size}x{grid_size}={total_slots}")
    
    fig = plt.figure(figsize=(25, 25))
    
    # Compute geometries and categorize points first
    geometries = []
    point_categories = []
    
    print(f"\nProcessing {n_squares} squares...")
    for i in range(n_squares):
        if i % 50 == 0:
            print(f"Processing square {i+1}...")
        geo = compute_square_geometry(squares[i])
        geometries.append(geo)
        cats = categorize_points(geo)
        point_categories.append(cats)
    
    # Plot each square
    for i in range(n_squares):
        ax = fig.add_subplot(grid_size, grid_size, i+1, projection='3d', computed_zorder=False)
        
        # Get categorized points
        cats = point_categories[i]
        geo = geometries[i]
        
        # Plot points with minimal size
        if len(cats['vertex_points']) > 0:
            ax.scatter(cats['vertex_points'][:, 0], cats['vertex_points'][:, 1], cats['vertex_points'][:, 2], 
                      c='red', s=1, zorder=4)
        if len(cats['face_points']) > 0:
            ax.scatter(cats['face_points'][:, 0], cats['face_points'][:, 1], cats['face_points'][:, 2], 
                      c='orange', s=1, zorder=3)
        if len(cats['interior_points']) > 0:
            ax.scatter(cats['interior_points'][:, 0], cats['interior_points'][:, 1], cats['interior_points'][:, 2], 
                      c='blue', s=1, zorder=2)
        
        # Plot hull faces with minimal opacity
        faces = geo['hull_faces']
        poly3d = [[face[0], face[1], face[2]] for face in faces]
        collection = Poly3DCollection(poly3d, 
                                    alpha=0.1,  # Reduced opacity
                                    linewidth=0.5,  # Thinner lines
                                    edgecolor='black',
                                    zorder=1)
        
        light_blue = colors.to_rgba('lightblue', alpha=0.1)  # Reduced opacity
        collection.set_facecolor([light_blue])
        collection.set_zsort('average')
        ax.add_collection3d(collection)
        
        # Minimal labels and formatting
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Set view angle
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1,1,1])
    
    plt.suptitle(f'Magic Square Convex Hulls (All {n_squares} Squares)', 
                 fontsize=16, y=0.95)
    plt.tight_layout()
    
    # Print summary statistics at the end
    vertex_counts = [len(cats['vertex_points']) for cats in point_categories]
    face_counts = [len(cats['face_points']) for cats in point_categories]
    interior_counts = [len(cats['interior_points']) for cats in point_categories]
    
    print("\nSummary Statistics:")
    print(f"Vertices: {np.mean(vertex_counts):.1f} ± {np.std(vertex_counts):.1f}")
    print(f"Face points: {np.mean(face_counts):.1f} ± {np.std(face_counts):.1f}")
    print(f"Interior points: {np.mean(interior_counts):.1f} ± {np.std(interior_counts):.1f}")
    
    # Count distribution patterns
    patterns = {}
    for v, f, i in zip(vertex_counts, face_counts, interior_counts):
        key = (v, f, i)
        patterns[key] = patterns.get(key, 0) + 1
    
    print("\nDistribution Patterns:")
    print("Vertices | Face Points | Interior | Count")
    print("-" * 45)
    for (v, f, i), count in sorted(patterns.items()):
        print(f"{v:8d} | {f:11d} | {i:8d} | {count:5d}")
    
    plt.show()

def plot_tsne_squares(squares, use_geometry=True, perplexity=30, n_iter=1000):
    """Plot t-SNE visualization of the magic squares"""
    
    print("\nComputing geometric properties...")
    # First compute all geometric properties
    geometries = []
    features = []
    for i, square in enumerate(squares):
        if i % 100 == 0:
            print(f"Processing square {i+1}/{len(squares)}...")
        
        geo = compute_square_geometry(square)
        cats = categorize_points(geo)
        n_vertices = len(cats['vertex_points'])
        n_faces = len(cats['face_points'])
        n_interior = len(cats['interior_points'])
        geometries.append((n_vertices, n_faces, n_interior))
        
        if use_geometry:
            # Use geometric features for t-SNE
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
    
    if use_geometry:
        # Use geometric features
        X = np.array(features)
        print(f"\nComputing t-SNE using geometric features...")
        print(f"Input shape: {X.shape} (n_squares × n_geometric_features)")
    else:
        # Use raw square values
        X = np.array(squares).reshape(len(squares), -1)
        print(f"\nComputing t-SNE using raw square values...")
        print(f"Input shape: {X.shape} (n_squares × 16)")
    
    # Standardize the features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    tsne = TSNE(n_components=2, 
                perplexity=perplexity, 
                n_iter=n_iter, 
                random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Color by number of vertices
    vertices = [g[0] for g in geometries]
    sc1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                     c=vertices, 
                     cmap='plasma', 
                     s=10, 
                     alpha=0.6)
    ax1.set_title('Colored by Number of Vertices')
    plt.colorbar(sc1, ax=ax1)
    
    # Plot 2: Color by number of face points
    faces = [g[1] for g in geometries]
    sc2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                     c=faces, 
                     cmap='plasma', 
                     s=10, 
                     alpha=0.6)
    ax2.set_title('Colored by Number of Face Points')
    plt.colorbar(sc2, ax=ax2)
    
    # Plot 3: Color by number of interior points
    interior = [g[2] for g in geometries]
    sc3 = ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                     c=interior, 
                     cmap='plasma', 
                     s=10, 
                     alpha=0.6)
    ax3.set_title('Colored by Number of Interior Points')
    plt.colorbar(sc3, ax=ax3)
    
    feature_type = "Geometric" if use_geometry else "Raw"
    plt.suptitle(f't-SNE Visualization of Magic Squares Using {feature_type} Features\nColored by Geometric Properties', 
                 fontsize=14)
    plt.tight_layout()
    
    # Print statistics
    print("\nt-SNE Statistics:")
    print(f"Feature type: {feature_type}")
    print(f"Perplexity: {perplexity}")
    print(f"Iterations: {n_iter}")
    print(f"Output shape: {X_tsne.shape}")
    
    # Print range of coordinates
    print("\nEmbedding Ranges:")
    print(f"X range: [{X_tsne[:,0].min():.2f}, {X_tsne[:,0].max():.2f}]")
    print(f"Y range: [{X_tsne[:,1].min():.2f}, {X_tsne[:,1].max():.2f}]")
    
    plt.show()
    
    return X_tsne, geometries

def plot_eigenvectors(squares):
    """Plot the first three eigenvectors for all squares with validation"""
    print("\nComputing eigenvectors for all squares...")
    
    # Collect eigenvectors, values, and covariance matrices
    eigendata = []
    cov_matrices = []
    
    for i, square in enumerate(squares):
        if i % 100 == 0:
            print(f"Processing square {i+1}/{len(squares)}...")
        geo = compute_square_geometry(square)
        if geo is not None:
            eigendata.append({
                'values': geo['eigenvalues'],
                'vectors': geo['eigenvectors'],
                'vertices': len(categorize_points(geo)['vertex_points'])
            })
            cov_matrices.append(geo['cov_matrix'])
    
    # Convert to numpy arrays for analysis
    cov_matrices = np.array(cov_matrices)
    vectors = np.array([d['vectors'] for d in eigendata])
    values = np.array([d['values'] for d in eigendata])
    
    # Print analysis
    print("\nCovariance Matrix Statistics:")
    print("Mean covariance matrix:")
    print(np.mean(cov_matrices, axis=0))
    print("\nStd of covariance matrix elements:")
    print(np.std(cov_matrices, axis=0))
    
    print("\nEigenvalue Statistics:")
    print("Mean eigenvalues:", np.mean(values, axis=0))
    print("Std eigenvalues:", np.std(values, axis=0))
    
    print("\nEigenvector Statistics:")
    for i in range(3):
        print(f"\nEigenvector {i+1}:")
        print(f"Mean direction: {np.mean(vectors[:,:,i], axis=0)}")
        print(f"Direction std:  {np.std(vectors[:,:,i], axis=0)}")
        
        # Check orthogonality
        if i < 2:
            dots = np.abs([np.dot(v[:,i], v[:,i+1]) for v in vectors])
            print(f"Mean dot product with next vector: {np.mean(dots):.6f}")
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot first component of each eigenvector
    scatter = ax.scatter(vectors[:,0,0], vectors[:,0,1], vectors[:,0,2],
                        c=[d['vertices'] for d in eigendata],
                        cmap='plasma',
                        s=10,
                        alpha=0.6)
    
    ax.set_xlabel('First Component')
    ax.set_ylabel('Second Component')
    ax.set_zlabel('Third Component')
    plt.colorbar(scatter, label='Number of Vertices')
    
    plt.title('First Components of Three Principal Eigenvectors\nColored by Number of Vertices')
    plt.show()
    
    return eigendata, cov_matrices
    # Add statistics
    eigenvalues = np.array([d['values'] for d in eigendata])
    print("\nEigenvalue Statistics:")
    print("Mean eigenvalues:", np.mean(eigenvalues, axis=0))
    print("Std eigenvalues:", np.std(eigenvalues, axis=0))
    
    # Print ranges
    print("\nEigenvector Component Ranges:")
    print(f"EV1 range: [{ev1[:,0].min():.3f}, {ev1[:,0].max():.3f}]")
    print(f"EV2 range: [{ev2[:,0].min():.3f}, {ev2[:,0].max():.3f}]")
    print(f"EV3 range: [{ev3[:,0].min():.3f}, {ev3[:,0].max():.3f}]")
    
    plt.show()
    
    # Also create pairwise plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # EV1 vs EV2
    scatter1 = ax1.scatter(ev1[:,0], ev2[:,0], 
                          c=vertices, cmap='plasma', s=10, alpha=0.6)
    ax1.set_xlabel('First Eigenvector')
    ax1.set_ylabel('Second Eigenvector')
    plt.colorbar(scatter1, ax=ax1)
    
    # EV2 vs EV3
    scatter2 = ax2.scatter(ev2[:,0], ev3[:,0], 
                          c=vertices, cmap='plasma', s=10, alpha=0.6)
    ax2.set_xlabel('Second Eigenvector')
    ax2.set_ylabel('Third Eigenvector')
    plt.colorbar(scatter2, ax=ax2)
    
    # EV1 vs EV3
    scatter3 = ax3.scatter(ev1[:,0], ev3[:,0], 
                          c=vertices, cmap='plasma', s=10, alpha=0.6)
    ax3.set_xlabel('First Eigenvector')
    ax3.set_ylabel('Third Eigenvector')
    plt.colorbar(scatter3, ax=ax3)
    
    plt.suptitle('Pairwise Eigenvector Components\nColored by Number of Vertices', 
                 fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return eigendata

def plot_comprehensive_analysis(squares):
    """Plot comprehensive analysis of all geometric properties"""
    print("\nComputing geometric properties for all squares...")
    
    # Collect all properties
    properties = []
    eigenvalue_ratios = []  # Separate list for eigenvalue ratios
    
    for i, square in enumerate(squares):
        if i % 100 == 0:
            print(f"Processing square {i+1}/{len(squares)}...")
        
        geo = compute_square_geometry(square)
        if geo is not None:
            props = {
                'vertices': len(categorize_points(geo)['vertex_points']),
                'volume': geo['hull_volume'],
                'area': geo['hull_area'],
                'mean_dist': geo['shape_descriptors']['mean_distance'],
                'std_dist': geo['shape_descriptors']['std_distance'],
                'angular_unif': geo['shape_descriptors']['angular_uniformity'],
                'vol_area_ratio': geo['shape_descriptors']['volume_area_ratio'],
                'compactness': geo['shape_descriptors']['compactness']
            }
            properties.append(props)
            
            # Calculate eigenvalue ratios separately
            evals = geo['eigenvalues']
            ratios = [evals[1]/evals[0], evals[2]/evals[1]]
            eigenvalue_ratios.append(ratios)
    
    # Convert to DataFrame for easier analysis
    import pandas as pd
    df = pd.DataFrame(properties)
    eigenvalue_ratios = pd.DataFrame(eigenvalue_ratios, columns=['λ2/λ1', 'λ3/λ2'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3)
    
    # 1. Distribution of basic properties
    ax1 = fig.add_subplot(gs[0, 0])
    df[['volume', 'area']].boxplot(ax=ax1)
    ax1.set_title('Hull Volume and Area Distribution')
    
    # 2. Shape descriptor distributions
    ax2 = fig.add_subplot(gs[0, 1])
    df[['mean_dist', 'std_dist', 'angular_unif']].boxplot(ax=ax2)
    ax2.set_title('Shape Descriptor Distribution')
    
    # 3. Vertex count distribution
    ax3 = fig.add_subplot(gs[0, 2])
    df['vertices'].hist(ax=ax3, bins=20)
    ax3.set_title('Number of Vertices Distribution')
    
    # 4. Volume vs Area scatter
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(df['volume'], df['area'], 
                         c=df['vertices'], cmap='plasma', alpha=0.6)
    ax4.set_xlabel('Volume')
    ax4.set_ylabel('Area')
    plt.colorbar(scatter, ax=ax4, label='Vertices')
    
    # 5. Eigenvalue ratios
    ax5 = fig.add_subplot(gs[1, 1])
    eigenvalue_ratios.boxplot(ax=ax5)
    ax5.set_title('Eigenvalue Ratios')
    
    # 6. t-SNE of shape descriptors
    ax6 = fig.add_subplot(gs[1, 2])
    features = df.values
    
    # Standardize features
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding = tsne.fit_transform(features)
    
    scatter = ax6.scatter(embedding[:, 0], embedding[:, 1], 
                         c=df['vertices'], cmap='plasma', alpha=0.6)
    ax6.set_title('t-SNE of Shape Descriptors')
    plt.colorbar(scatter, ax=ax6, label='Vertices')
    
    # 7. Correlation matrix
    ax7 = fig.add_subplot(gs[2, :])
    corr = df.corr()
    im = ax7.imshow(corr, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax7)
    ax7.set_xticks(range(len(corr.columns)))
    ax7.set_yticks(range(len(corr.columns)))
    ax7.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax7.set_yticklabels(corr.columns)
    ax7.set_title('Correlation Matrix of Properties')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nEigenvalue Ratios Statistics:")
    print(eigenvalue_ratios.describe())
    
    # Print strongest correlations
    print("\nStrongest Property Correlations:")
    correlations = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            correlations.append((
                corr.columns[i],
                corr.columns[j],
                corr.iloc[i,j]
            ))
    
    for prop1, prop2, corr_val in sorted(correlations, key=lambda x: abs(x[2]), reverse=True)[:5]:
        print(f"{prop1} vs {prop2}: {corr_val:.3f}")
    
    return df, eigenvalue_ratios

def compute_emd_distance(square1, square2):
    """Compute Earth Mover's Distance between two magic squares"""
    # Convert squares into point clouds in 3D
    n = square1.shape[0]
    points1 = []
    points2 = []
    
    # Create 3D points (x, y, value)
    for i in range(n):
        for j in range(n):
            points1.append([i/n, j/n, square1[i,j]/34])  # Normalize coordinates
            points2.append([i/n, j/n, square2[i,j]/34])  # 34 is max value in 4x4 magic square
    
    points1 = np.array(points1)
    points2 = np.array(points2)
    
    # Create uniform weights for points
    weights1 = np.ones(len(points1)) / len(points1)
    weights2 = np.ones(len(points2)) / len(points2)
    
    # Compute cost matrix using Euclidean distance
    M = ot.dist(points1, points2, metric='euclidean')
    M /= M.max()  # Normalize costs
    
    # Compute EMD
    return ot.emd2(weights1, weights2, M)

def plot_tsne_emd(squares, perplexity=30, distance_threshold=None):
    """Create t-SNE visualization using Earth Mover's Distance with threshold-based connections"""
    print("\nComputing EMD distances between squares...")
    
    # Compute distance matrix
    n_squares = len(squares)
    distances = np.zeros((n_squares, n_squares))
    
    for i in range(n_squares):
        if i % 10 == 0:  # Progress indicator
            print(f"Processing square {i+1}/{n_squares}...")
        for j in range(i+1, n_squares):
            dist = compute_emd_distance(squares[i], squares[j])
            distances[i,j] = distances[j,i] = dist
    
    # If no threshold provided, use a percentile of the distances
    if distance_threshold is None:
        # Use the 5th percentile of non-zero distances as default threshold
        nonzero_distances = distances[distances > 0]
        distance_threshold = np.percentile(nonzero_distances, 1)
        print(f"\nUsing distance threshold: {distance_threshold:.4f} (5th percentile)")
    
    # Compute t-SNE embedding
    print("\nComputing t-SNE embedding...")
    tsne = TSNE(n_components=2, 
                metric='precomputed',
                perplexity=perplexity,
                init='random',
                random_state=42)
    embedding = tsne.fit_transform(distances)
    
    # Create visualization
    plt.figure(figsize=(15, 15))
    
    # First plot the connections under threshold
    print("\nFinding connections below threshold...")
    connection_count = 0
    for i in range(n_squares):
        for j in range(i+1, n_squares):
            if distances[i,j] < distance_threshold:
                plt.plot([embedding[i,0], embedding[j,0]],
                        [embedding[i,1], embedding[j,1]],
                        'gray', alpha=0.1, linewidth=0.5)
                connection_count += 1
    
    print(f"Found {connection_count} connections below threshold")
    
    # Compute geometric properties for coloring
    print("\nComputing geometric properties for visualization...")
    vertex_counts = []
    for square in squares:
        geo = compute_square_geometry(square)
        cats = categorize_points(geo)
        vertex_counts.append(len(cats['vertex_points']))
    
    # Main scatter plot
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                         c=vertex_counts,
                         cmap='plasma',
                         alpha=0.8,
                         s=50,
                         zorder=2)  # Ensure points are drawn on top of lines
    plt.colorbar(scatter, label='Number of Vertices')
    
    plt.title(f'Magic Square Similarity Network (EMD < {distance_threshold:.4f})\n'
              f'Perplexity={perplexity}, {connection_count} connections')
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics about the connections
    print("\nConnection Statistics:")
    connections_per_square = np.sum(distances < distance_threshold, axis=1) - 1  # -1 to exclude self
    print(f"Average connections per square: {np.mean(connections_per_square):.1f}")
    print(f"Min connections: {np.min(connections_per_square)}")
    print(f"Max connections: {np.max(connections_per_square)}")
    
    return distances, embedding, distance_threshold

def compute_all_emd_distances(squares, cache_file='emd_distances.pkl'):
    """Compute or load cached EMD distances between all squares"""
    
    # Create absolute path for cache file
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    cache_path = os.path.join(cache_dir, cache_file)
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"Loading cached EMD distances from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("\nComputing EMD distances between all squares...")
    n_squares = len(squares)
    distances = np.zeros((n_squares, n_squares))
    
    total_computations = (n_squares * (n_squares - 1)) // 2
    computation_count = 0
    
    for i in range(n_squares):
        if i % 10 == 0:
            progress = (computation_count / total_computations) * 100
            print(f"Processing square {i+1}/{n_squares}... ({progress:.1f}% complete)")
        for j in range(i+1, n_squares):
            dist = compute_emd_distance(squares[i], squares[j])
            distances[i,j] = distances[j,i] = dist
            computation_count += 1
    
    # Save to cache
    print(f"Saving EMD distances to {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(distances, f)
    
    return distances

def analyze_distance_distributions(magic_squares, n_random=1000, cache_file='emd_distances.pkl'):
    """Compare EMD distances between:
    1. Legal magic squares vs legal magic squares
    2. Permuted magic squares vs legal magic squares"""
    
    print("\nAnalyzing distance distributions...")
    
    # Load cached magic square distances
    magic_distances_matrix = compute_all_emd_distances(magic_squares, cache_file)
    magic_distances = magic_distances_matrix[np.triu_indices_from(magic_distances_matrix, k=1)]
    print(f"Using {len(magic_distances)} magic square distances from cache")
    
    # Generate permuted versions of magic squares
    print(f"\nComputing {n_random} permuted magic square comparisons...")
    permuted_vs_magic_distances = []
    
    for i in range(n_random):
        if i % 100 == 0:
            print(f"Processing permutation {i+1}/{n_random}...")
        
        # Select a random magic square and permute it
        original_idx = np.random.randint(0, len(magic_squares))
        original_square = magic_squares[original_idx]
        permuted_square = np.random.permutation(original_square.flatten()).reshape(4, 4)
        
        # Compare permuted square with all legal magic squares
        for j in range(len(magic_squares)):
            if j != original_idx:  # Skip self-comparison
                dist = compute_emd_distance(permuted_square, magic_squares[j])
                permuted_vs_magic_distances.append(dist)
    
    # Plot distributions
    plt.figure(figsize=(12, 6))
    
    plt.hist(magic_distances, bins=50, alpha=0.5, 
             label='Legal vs Legal', density=True, color='blue')
    plt.hist(permuted_vs_magic_distances, bins=50, alpha=0.5,
             label='Permuted vs Legal', density=True, color='red')
    
    plt.xlabel('Earth Mover\'s Distance')
    plt.ylabel('Density')
    plt.title('Distribution of EMD: Legal vs Permuted Magic Squares')
    plt.legend()
    
    # Add statistical test
    ks_stat, p_value = stats.ks_2samp(magic_distances, permuted_vs_magic_distances)
    
    print("\nStatistical Analysis:")
    print(f"KS test statistic: {ks_stat:.3f}")
    print(f"p-value: {p_value:.3e}")
    print(f"\nLegal vs Legal mean distance: {np.mean(magic_distances):.3f} ± {np.std(magic_distances):.3f}")
    print(f"Permuted vs Legal mean distance: {np.mean(permuted_vs_magic_distances):.3f} ± {np.std(permuted_vs_magic_distances):.3f}")
    
    # Add effect size
    cohens_d = (np.mean(magic_distances) - np.mean(permuted_vs_magic_distances)) / \
               np.sqrt((np.var(magic_distances) + np.var(permuted_vs_magic_distances)) / 2)
    print(f"Cohen's d effect size: {cohens_d:.3f}")
    
    plt.show()
    
    return magic_distances, permuted_vs_magic_distances, magic_distances_matrix


def plot_geometric_tsne_with_emd(squares, perplexity=30, distance_threshold=None):
    """Plot geometric t-SNE with EMD connections overlaid"""
    print("\nComputing geometric properties...")
    
    properties = []
    for i, square in enumerate(squares):
        if i % 100 == 0:
            print(f"Processing square {i+1}/{len(squares)}...")
        geo = compute_square_geometry(square)
        if geo is not None:
            props = {
                'vertices': len(categorize_points(geo)['vertex_points']),
                'volume': geo['hull_volume'],
                'area': geo['hull_area'],
                'mean_dist': geo['shape_descriptors']['mean_distance'],
                'std_dist': geo['shape_descriptors']['std_distance'],
                'angular_unif': geo['shape_descriptors']['angular_uniformity'],
                'vol_area_ratio': geo['shape_descriptors']['volume_area_ratio'],
                'compactness': geo['shape_descriptors']['compactness']
            }
            properties.append(props)
    
    features = np.array([[p[key] for key in props.keys()] for p in properties])
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    
    print("\nComputing geometric t-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedding = tsne.fit_transform(features)
    
    # Load or compute EMD distances
    distances = compute_all_emd_distances(squares)
    
    if distance_threshold is None:
        nonzero_distances = distances[distances > 0]
        distance_threshold = np.percentile(nonzero_distances, 5)
        print(f"\nUsing distance threshold: {distance_threshold:.4f} (5th percentile)")
    
    plt.figure(figsize=(15, 15))
    
    print("\nPlotting connections...")
    connection_count = 0
    for i in range(len(squares)):
        for j in range(i+1, len(squares)):
            if distances[i,j] < distance_threshold:
                plt.plot([embedding[i,0], embedding[j,0]],
                        [embedding[i,1], embedding[j,1]],
                        'gray', alpha=0.1, linewidth=0.5)
                connection_count += 1
    
    vertex_counts = [p['vertices'] for p in properties]
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                         c=vertex_counts,
                         cmap='plasma',
                         alpha=0.8,
                         s=50,
                         zorder=2)
    plt.colorbar(scatter, label='Number of Vertices')
    
    plt.title(f'Geometric t-SNE with EMD Connections (EMD < {distance_threshold:.4f})\n'
              f'Perplexity={perplexity}, {connection_count} connections')
    
    plt.tight_layout()
    plt.show()
    
    return embedding, distances, distance_threshold

def estimate_manifold_dimension(distances, methods=['correlation', 'mle', 'nearest_neighbor']):
    """Estimate the intrinsic dimension of the magic square manifold using multiple methods"""
    print("\nEstimating manifold dimension...")
    results = {}
    
    if 'correlation' in methods:
        print("\nComputing Correlation Dimension...")
        r_values = np.logspace(-1, 2, 100)
        correlations = []
        
        for r in r_values:
            correlations.append(np.sum(distances < r) / (len(distances) * len(distances)))
        
        correlations = np.array(correlations)
        valid_idx = np.where((correlations > 0.01) & (correlations < 0.9))[0]
        
        if len(valid_idx) > 10:
            slope, intercept, r_value, _, _ = linregress(
                np.log(r_values[valid_idx]),
                np.log(correlations[valid_idx])
            )
            results['correlation'] = {
                'dimension': slope,
                'r_squared': r_value**2
            }
            
            # Plot correlation dimension estimation
            plt.figure(figsize=(10, 6))
            plt.loglog(r_values, correlations, 'b.', label='Data')
            # Plot the fitted line
            fit_y = np.exp(slope * np.log(r_values[valid_idx]) + intercept)
            plt.loglog(r_values[valid_idx], fit_y, 'r-', 
                      label=f'Fit (d ≈ {slope:.2f}, R² = {r_value**2:.3f})')
            plt.xlabel('Radius (r)')
            plt.ylabel('C(r)')
            plt.title('Correlation Dimension Estimation')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.legend()
            plt.show()
    
    if 'nearest_neighbor' in methods:
        print("\nComputing k-NN Dimension...")
        k_values = np.arange(1, 21)  # Test k=1 to k=20
        dimensions = []
        
        for k in k_values:
            # Sort distances for each point
            sorted_dists = np.sort(distances, axis=1)
            
            # Use k and k/2 nearest neighbors
            r_k = sorted_dists[:, k]
            r_k2 = sorted_dists[:, k//2] if k > 1 else sorted_dists[:, 1]
            
            # Compute ratios and handle edge cases
            ratios = r_k / r_k2
            # Filter out problematic ratios (too close to 1 or invalid)
            valid_ratios = (ratios > 1.0001) & (ratios < np.inf)
            
            if np.any(valid_ratios):
                # Compute dimension estimates only for valid ratios
                dims = np.log(2) / np.log(ratios[valid_ratios])
                # Filter out unrealistic values
                valid_dims = dims[(dims > 0) & (dims < 50)]  # Upper bound lowered
                
                if len(valid_dims) > 0:
                    # Use median to reduce sensitivity to outliers
                    dimensions.append(np.median(valid_dims))
        
        if dimensions:
            results['knn'] = {
                'dimension_min': np.min(dimensions),
                'dimension_max': np.max(dimensions),
                'dimension_median': np.median(dimensions)
            }
            
            # Plot k-NN dimension estimates
            plt.figure(figsize=(10, 6))
            plt.plot(k_values[:len(dimensions)], dimensions, 'b.-')
            plt.xlabel('k (number of neighbors)')
            plt.ylabel('Estimated Dimension')
            plt.title('k-NN Dimension Estimates')
            plt.grid(True)
            plt.show()
    
    return results

# Add to your analysis:
def analyze_manifold(magic_squares, cache_file='emd_distances.pkl'):
    """Analyze the manifold of magic squares"""
    
    # Load cached distances
    distances = compute_all_emd_distances(magic_squares, cache_file)
    
    # Estimate manifold dimension
    dimension_estimates = estimate_manifold_dimension(distances)
    
    return dimension_estimates
