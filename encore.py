import numpy as np
import matplotlib.pyplot as plt

# Function to load an .off file and extract both vertices and faces (for mesh rendering)
def load_off_with_faces(filename):
    with open(filename, 'r') as f:
        # Read the header
        header = f.readline().strip()
        if 'OFF' != header:
            raise Exception("Not a valid OFF header")

        # Read the number of vertices, faces, and edges
        n_vertices, n_faces, n_edges = map(int, f.readline().strip().split())

        # Read the vertices
        vertices = []
        for _ in range(n_vertices):
            vertex = list(map(float, f.readline().strip().split()))
            vertices.append(vertex)

        # Read the faces
        faces = []
        for _ in range(n_faces):
            face = list(map(int, f.readline().strip().split()))[1:]  # Skip the face count (first number)
            faces.append(face)

        # Convert to numpy arrays for easy manipulation
        vertices = np.array(vertices)
        faces = np.array(faces)

    return vertices, faces


# Load the vertices and faces from the .off file
vertices, faces = load_off_with_faces("/home/jean-louis/Documents/Code/ModelNet10/bed/train/bed_0001.off")

# Check the shape of faces to confirm correct loading
vertices.shape, faces.shape

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Create the 3D figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

z_values = vertices[:, 2]

z_min, z_max = np.min(z_values), np.max(z_values)

# Normalize the Z values to [0, 1] for gradient coloring
norm_z = (vertices[:, 2] - z_min) / (z_max - z_min)
face_colors = plt.cm.viridis(norm_z)

# Create the polygon collection for the faces, colored by Z value
mesh = Poly3DCollection(vertices[faces], facecolors=face_colors[faces].mean(axis=1), linewidths=0.1, edgecolors='k')

# Add the mesh to the plot
ax.add_collection3d(mesh)

# Set plot limits
ax.set_xlim(np.min(vertices[:, 0]), np.max(vertices[:, 0]))
ax.set_ylim(np.min(vertices[:, 1]), np.max(vertices[:, 1]))
ax.set_zlim(np.min(vertices[:, 2]), np.max(vertices[:, 2]))

# Set plot labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Mesh of ModelNet Bed with Z-axis Gradient Color')

# Adjust the view angle for better visualization
ax.view_init(elev=20, azim=60)

plt.show()
