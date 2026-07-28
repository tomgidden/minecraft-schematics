#!/usr/bin/env python3
"""
Geodesic Dome Builder for Minecraft using Minescript 5

Credits:
- Geodesic dome tessellation algorithm from 'geodome' package by nbhr
  https://github.com/nbhr/geodome
  Copyright (c) 2020 nbhr, MIT License

Usage: Run this script from within Minecraft with Minescript installed.
"""

import sys
import numpy as np
import minescript as ms

# Dome shape parameters
RADIUS = 32  # Base dome radius in blocks
SUBDIVISIONS = 1  # Tessellation level (0-4, higher = more detail but slower)

# Ellipsoid scaling: 1.0 = sphere, different values create ellipsoid shapes
SCALE_X = 1.0  # Width scale factor (east-west in Minecraft)
SCALE_Y = 1.0  # Height scale factor (vertical in Minecraft)
SCALE_Z = 1.0  # Depth scale factor (north-south in Minecraft)

# Hemisphere option: Only build the top half (above and including equator)
HEMISPHERE_ONLY = True  # Set to True to only build top half

# Edge parameters
EDGE_THICKNESS_RADIAL = 1  # Thickness in radial direction (toward/away from centre)
EDGE_THICKNESS_TANGENTIAL = 1  # Thickness tangent to surface, perpendicular to edge
DO_EDGES = True  # Set to False to see face-only without edges

# Face/glass parameters
DO_FACES = True  # Fill faces with glass (True) or edges only (False)
CURVED_FACES = True  # If True: smooth ellipsoid shell; False: flat triangular panels
FACE_INSET = 0.5  # How many blocks to inset the face blocks from outer edge surface

DO_VERTICES = False  # Mark edge vertices with marker blocks (useful when DO_EDGES=False)
ONLY_IN_AIR = False  # Only place blocks in air (slow), rather than overwrite everything

CUTOFF_LAYERS = 0  # Remove bottom N layers (None = full dome, or specify number)

# Block materials
EDGE_BLOCK = "stone_bricks"
FACE_BLOCK = "light_gray_stained_glass"
VERTEX_MARKER_BLOCK = "red_stained_glass"


class GeodesicDome:
    """
    Geodesic Dome mesh generator.
    Creates an icosahedron and subdivides it to create geodesic dome geometry.

    Attributes:
        v: (Nv, 3) numpy array of vertices
        f: (Nf, 3) numpy array of face indices (triangles)

    Original implementation by nbhr (https://github.com/nbhr/geodome)
    """

    def __init__(self):
        """Create initial icosahedron (Level 0 geodesic dome)."""
        # Calculate icosahedron geometry using golden ratio
        p = (1 + np.sqrt(5)) / 2  # Golden ratio
        a = np.sqrt((3 + 4 * p) / 5) / 2
        b = np.sqrt(p / np.sqrt(5))
        c = np.sqrt(3 / 4 - a**2)
        d = np.sqrt(3 / 4 - (b - a) ** 2)

        # Define 12 vertices in cylindrical coordinates (r, theta, z)
        self.v = np.array(
            [
                [0, 0, (c + d / 2)],
                [b, 2 * 0 * np.pi / 5, d / 2],
                [b, 2 * 1 * np.pi / 5, d / 2],
                [b, 2 * 2 * np.pi / 5, d / 2],
                [b, 2 * 3 * np.pi / 5, d / 2],
                [b, 2 * 4 * np.pi / 5, d / 2],
                [b, (2 * 0 + 1) * np.pi / 5, -d / 2],
                [b, (2 * 1 + 1) * np.pi / 5, -d / 2],
                [b, (2 * 2 + 1) * np.pi / 5, -d / 2],
                [b, (2 * 3 + 1) * np.pi / 5, -d / 2],
                [b, (2 * 4 + 1) * np.pi / 5, -d / 2],
                [0, 0, -(c + d / 2)],
            ]
        )

        # Convert from cylindrical to Cartesian coordinates
        # Map to Minecraft coords: x=x, y=z(vertical), z=y(depth)
        self.v = np.vstack(
            [
                self.v[:, 0] * np.cos(self.v[:, 1]),  # x: east-west
                self.v[:, 2],  # y: vertical (was z in cylindrical)
                self.v[:, 0] * np.sin(self.v[:, 1]),  # z: north-south (was y in cylindrical)
            ]
        ).T

        # Normalize to unit sphere (using y coordinate which is now vertical)
        self.v *= 1 / self.v[0, 1]

        # Clean up near-zero values
        self.tol = 1e-15
        self.v[np.abs(self.v) < self.tol] = 0

        # Define 20 triangular faces of icosahedron
        self.f = np.array(
            [
                [2, 0, 1],
                [3, 0, 2],
                [4, 0, 3],
                [5, 0, 4],
                [1, 0, 5],
                [2, 1, 6],
                [7, 2, 6],
                [3, 2, 7],
                [8, 3, 7],
                [4, 3, 8],
                [9, 4, 8],
                [5, 4, 9],
                [10, 5, 9],
                [6, 1, 10],
                [1, 5, 10],
                [6, 11, 7],
                [7, 11, 8],
                [8, 11, 9],
                [9, 11, 10],
                [10, 11, 6],
            ]
        )

    def tessellate(self, iter=1):
        """
        Subdivide the geodesic dome mesh.
        Each iteration subdivides each triangle into 4 sub-triangles.
        """

        def newvert(v0, v1):
            """Create new vertex at midpoint of edge, projected to unit sphere."""
            v = v0 + v1
            v /= np.linalg.norm(v)
            return v

        for _ in range(iter):
            f = self.f
            v = self.v
            v2 = []
            vv2v = {}  # Maps edge (vertex pair) to new vertex ID
            vid = len(v)

            # Create new vertices at edge midpoints
            for tri in self.f:
                for i, j in zip([0, 1, 2], [1, 2, 0]):
                    if tri[i] < tri[j]:
                        vv2v[tri[i], tri[j]] = vv2v[tri[j], tri[i]] = vid
                        vid += 1
                        v2.append(newvert(v[tri[i]], v[tri[j]]))

            v = np.vstack([v, np.array(v2)])

            # Subdivide each triangle into 4 sub-triangles
            f2 = []
            for tri in self.f:
                # Three corner triangles
                f2.append([tri[0], vv2v[tri[0], tri[1]], vv2v[tri[2], tri[0]]])
                f2.append([tri[1], vv2v[tri[1], tri[2]], vv2v[tri[0], tri[1]]])
                f2.append([tri[2], vv2v[tri[2], tri[0]], vv2v[tri[1], tri[2]]])
                # centre triangle
                f2.append([vv2v[tri[0], tri[1]], vv2v[tri[1], tri[2]], vv2v[tri[2], tri[0]]])

            self.v = v
            self.f = np.array(f2)

        # Clean up near-zero values
        self.v[np.abs(self.v) < self.tol] = 0
        return self


class MinecraftGeodesicDome:
    def __init__(
        self,
        centre_x,  # Centre position of the dome
        centre_y,  # Centre position of the dome
        centre_z,  # Centre position of the dome
        radius,  # Base radius of the dome in blocks (not including 1 for centre row/column)
        subdivisions,  # Number of tessellation subdivisions (0-4 recommended)
        edge_thickness_radial,  # Thickness in radial direction (toward/away from centre)
        edge_thickness_tangential,  # Thickness tangent to surface, perpendicular to edge
        scale_x=1.0,  # Scale factor for X axis (east-west, 1.0 = sphere)
        scale_y=1.0,  # Scale factor for Y axis (vertical, 1.0 = sphere)
        scale_z=1.0,  # Scale factor for Z axis (north-south, 1.0 = sphere)
        cutoff_layers=None,  # Number of bottom layers to remove (None = full dome)
        do_faces=True,  # Whether to fill faces with glass (True) or only draw edges (False)
        do_edges=True,  # Whether to draw edges (True) or only fill faces (False)
        do_vertices=False,  # Whether to draw vertices (True) or not (False)
        hemisphere_only=False,  # Whether to draw only the upper hemisphere (True) or both hemispheres (False)
        face_inset=0.0,  # Inset of face blocks from the edge of the dome
        curved_faces=False,  # Whether to use curved faces (True) or flat faces (False)
        only_in_air=False,  # Whether to only place blocks in air (True), or overwrite all blocks (False)
        edge_block=None,  # Block to use for edges
        face_block=None,  # Block to use for faces
        vertex_marker_block=None,  # Block to use for vertex markers
    ):
        self.centre = np.array([centre_x, centre_y, centre_z])
        self.radius = radius
        self.subdivisions = subdivisions
        self.edge_thickness_radial = edge_thickness_radial
        self.edge_thickness_tangential = edge_thickness_tangential
        self.cutoff_layers = cutoff_layers
        self.do_faces = do_faces
        self.do_edges = do_edges
        self.do_vertices = do_vertices
        self.scale = np.array([scale_x, scale_y, scale_z])
        self.hemisphere_only = hemisphere_only
        self.face_inset = face_inset
        self.curved_faces = curved_faces
        self.only_in_air = only_in_air

        if edge_block is None:
            # No edge blocks specified, so disable edges
            self.do_edges = False
        else:
            # Edge blocks specified, so do as specified by do_edges parameter
            self.do_edges = do_edges
            self.edge_block = edge_block

        if face_block is None:
            # No face blocks specified, so disable faces
            self.do_faces = False
        else:
            # Face blocks specified, so do as specified by do_faces parameter
            self.do_faces = do_faces
            self.face_block = face_block

        if vertex_marker_block is None:
            # No vertex marker blocks specified, so disable vertices
            self.do_vertices = False
        else:
            # Vertex marker blocks specified, so do as specified by do_vertices parameter
            self.do_vertices = do_vertices
            self.vertex_marker_block = vertex_marker_block

        # Generate the geodesic dome geometry
        self.dome = GeodesicDome()
        self.dome.tessellate(self.subdivisions)

        # Scale vertices to desired radius and apply ellipsoid scaling
        # Add 0.5 to radius for symmetric block placement with floor() rounding
        # (centre at 0.5, 0, 0.5 requires radius+0.5 to reach blocks ±radius)
        effective_radius = radius + 0.5
        self.vertices = self.dome.v * effective_radius

        # Apply per-axis scaling for ellipsoid
        self.vertices = self.vertices * self.scale

        # Create inner vertices for faces (inset from outer surface)
        self.inner_vertices = self._create_inner_vertices()

        # Apply hemisphere filter if specified (before cutoff)
        if hemisphere_only:
            self._apply_hemisphere_filter()

        # Apply cutoff if specified
        if cutoff_layers is not None:
            self._apply_cutoff()

    def _create_inner_vertices(self):
        """
        Create inner vertices for glass placement by insetting from outer surface.
        Each vertex is moved inward along its normal direction (toward centre).
        """
        inner_vertices = np.zeros_like(self.vertices)

        for i, vertex in enumerate(self.vertices):
            # Calculate the direction from centre to vertex (normal direction)
            # Account for ellipsoid scaling
            vertex_normalized = vertex / self.scale
            normal = vertex_normalized / np.linalg.norm(vertex_normalized)

            # Move inward by face_inset amount, accounting for ellipsoid scaling
            inset_vector = normal * self.face_inset * self.scale
            inner_vertices[i] = vertex - inset_vector

        return inner_vertices

    def _apply_cutoff(self):
        """Remove bottom layers of the dome based on cutoff_layers parameter."""
        # Calculate y threshold (y is vertical in Minecraft)
        y_values = self.vertices[:, 1]
        min_y = y_values.min()
        max_y = y_values.max()
        y_range = max_y - min_y

        # Calculate cutoff threshold
        # Use effective radius for consistency with vertex scaling
        effective_radius = self.radius + 0.5
        cutoff_threshold = min_y + (y_range * self.cutoff_layers / effective_radius)

        # Filter faces that have all vertices above threshold
        face_mask = np.all(self.vertices[self.dome.f][:, :, 1] >= cutoff_threshold, axis=1)
        self.dome.f = self.dome.f[face_mask]

    def _apply_hemisphere_filter(self):
        """Keep only the top half of the dome (above equator, y >= 0)."""
        # Filter faces that have all vertices at or above the equator (y >= 0)
        face_mask = np.all(self.vertices[self.dome.f][:, :, 1] >= 0, axis=1)
        self.dome.f = self.dome.f[face_mask]

    def _get_block_coords(self, x, y, z):
        return (
            int(round(x + self.centre[0])),
            int(round(y + self.centre[1])),
            int(round(z + self.centre[2])),
        )

    def _should_place_block(self, xyz, block_region=None):
        """Check if we should place a block at this position (only in air)."""
        if not self.only_in_air:
            return True

        if block_region is not None:
            # Use cached block region (FAST!)
            try:
                current_block = block_region.get_block(xyz[0], xyz[1], xyz[2])
                # BlockRegion returns None for air blocks, but might also have explicit air strings
                if current_block is None:
                    return True
                # Check for explicit air block types (with or without minecraft: prefix)
                if current_block in [
                    "air",
                    "cave_air",
                    "void_air",
                    "minecraft:air",
                    "minecraft:cave_air",
                    "minecraft:void_air",
                ]:
                    return True
                return False
            except IndexError:
                # Outside cached region, fall back to direct call
                current_block = ms.getblock(xyz[0], xyz[1], xyz[2])
        else:
            # No cache available (SLOW!)
            current_block = ms.getblock(xyz[0], xyz[1], xyz[2])

        # Handle direct getblock() results (always includes minecraft: prefix)
        return current_block in ["minecraft:air", "minecraft:cave_air", "minecraft:void_air"]

    def _draw_edge_with_tracking(self, v1, v2, packer, block_region=None):
        """
        Draw an edge between two vertices with radial and tangential thickness.
        Radial thickness: toward/away from centre
        Tangential thickness: perpendicular to edge, tangent to surface

        Returns: Set of (x, y, z) tuples for all placed blocks
        """
        placed_blocks = set()

        # Interpolate points along the edge with higher resolution
        edge_length = np.linalg.norm(v2 - v1)
        num_points = int(edge_length * 3) + 2  # Higher resolution for smoothing

        # Edge direction vector
        edge_dir = (v2 - v1) / edge_length if edge_length > 0 else np.array([0, 0, 1])

        for i in range(num_points):
            t = i / max(num_points - 1, 1)
            point = v1 * (1 - t) + v2 * t

            # For ellipsoids, project back onto the ellipsoid surface
            # Use effective_radius to match vertex scaling
            effective_radius = self.radius + 0.5
            point_normalized = point / self.scale / effective_radius
            point_norm = np.linalg.norm(point_normalized)
            if point_norm > 0.01:
                point_normalized = point_normalized / point_norm

            # Calculate radial direction (from centre toward point)
            radial_dir_normalized = point_normalized if point_norm > 0.01 else np.array([0, 0, 1])

            # Calculate tangent perpendicular direction (perpendicular to both edge and radial)
            tangent_perp = np.cross(edge_dir, radial_dir_normalized)
            tangent_perp_norm = np.linalg.norm(tangent_perp)
            if tangent_perp_norm > 0.01:
                tangent_perp = tangent_perp / tangent_perp_norm
            else:
                # Edge is radial, use arbitrary perpendicular
                tangent_perp = (
                    np.array([1, 0, 0])
                    if abs(radial_dir_normalized[0]) < 0.9
                    else np.array([0, 1, 0])
                )
                tangent_perp = np.cross(radial_dir_normalized, tangent_perp)
                tangent_perp = tangent_perp / np.linalg.norm(tangent_perp)

            # Apply radial thickness: sample at multiple radii
            radial_samples = max(1, int(self.edge_thickness_radial * 2 + 1))
            radial_offsets = np.linspace(
                -self.edge_thickness_radial, self.edge_thickness_radial, radial_samples
            )

            # Apply tangential thickness: sample perpendicular to edge
            tangent_samples = max(1, int(self.edge_thickness_tangential * 2 + 1))
            tangent_offsets = np.linspace(
                -self.edge_thickness_tangential,
                self.edge_thickness_tangential,
                tangent_samples,
            )

            for radial_offset in radial_offsets:
                # Scale the point radially (in ellipsoid-normalized space)
                scale_factor = 1.0 + radial_offset / effective_radius
                point_at_radius = point_normalized * scale_factor * effective_radius * self.scale

                for tangent_offset in tangent_offsets:
                    # Offset perpendicular to edge, tangent to surface
                    offset_point = point_at_radius + tangent_perp * tangent_offset * self.scale

                    # Get block coordinates; use floor for consistent rounding (matches edge drawing behavior)
                    xyz = (
                        int(np.floor(offset_point[0] + self.centre[0])),
                        int(np.floor(offset_point[1] + self.centre[1])),
                        int(np.floor(offset_point[2] + self.centre[2])),
                    )

                    # Only place if not already placed and current block is air
                    if xyz not in placed_blocks and self._should_place_block(xyz, block_region):
                        block = self.edge_block
                        packer.setblock(xyz, block)
                        placed_blocks.add(xyz)

        return placed_blocks

    def _create_ellipsoid_shell(self, packer, block_region=None):
        """
        Create a smooth ellipsoid shell of glass using spherical sampling.
        The shell is inset from the outer surface by face_inset amount.

        Returns: Set of (x, y, z) tuples where face blocks were placed
        """

        # Calculate inner radius (inset from outer surface)
        # For symmetric block placement with centre at (0.5, 0, 0.5):
        # - Block ±R has centre at (±R+0.5, y, ±R+0.5)
        # - Distance from (0.5,0,0.5) to (R+0.5, y, R+0.5) is R
        # - But floor() truncates, so we need radius + 0.5 to reach both ±R blocks
        inner_radius = self.radius + 0.5 - self.face_inset

        # High-resolution spherical sampling to ensure no gaps
        # Sample density: ensure adjacent samples are < 0.3 blocks apart on the surface
        max_scaled_radius = inner_radius * max(self.scale[0], self.scale[1], self.scale[2])

        # Azimuthal angle (around equator): full circle
        theta_samples = max(100, int(2 * np.pi * max_scaled_radius / 0.3))

        # Polar angle (from top pole to bottom pole or top pole to equator)
        if self.hemisphere_only:
            # Hemisphere: 0 to π/2 (top half)
            phi_samples = max(50, int(np.pi * max_scaled_radius / 2 / 0.3))
            phi_range = np.linspace(0, np.pi / 2, phi_samples)
        else:
            # Full sphere: 0 to π
            phi_samples = max(100, int(np.pi * max_scaled_radius / 0.3))
            phi_range = np.linspace(0, np.pi, phi_samples)

        theta_range = np.linspace(0, 2 * np.pi, theta_samples, endpoint=False)

        ms.echo(
            f"Creating ellipsoid shell ({theta_samples}×{phi_samples} samples = {theta_samples * phi_samples} points)..."
        )

        placed_blocks = set()
        sample_count = 0

        for phi in phi_range:
            for theta in theta_range:
                # Convert spherical coordinates to Cartesian (y-vertical)
                x = inner_radius * np.sin(phi) * np.cos(theta) * self.scale[0]
                y = inner_radius * np.cos(phi) * self.scale[1]
                z = inner_radius * np.sin(phi) * np.sin(theta) * self.scale[2]

                # Convert to block coordinates
                # Use floor for consistent rounding (matches edge drawing behavior)
                xyz = (
                    int(np.floor(x + self.centre[0])),
                    int(np.floor(y + self.centre[1])),
                    int(np.floor(z + self.centre[2])),
                )

                # Place glass (set automatically handles duplicates)
                if xyz not in placed_blocks and self._should_place_block(xyz, block_region):
                    packer.setblock(xyz, self.face_block)
                    placed_blocks.add(xyz)

                sample_count += 1
                if sample_count % 10000 == 0:
                    ms.echo(
                        f"  Sampled {sample_count}/{theta_samples * phi_samples} points, placed {len(placed_blocks)} blocks"
                    )

        ms.echo(f"Placed {len(placed_blocks)} face blocks for shell")
        return placed_blocks

    def _fill_face(
        self,
        face_vertices: list[int],
        packer: ms.BlockPacker,
        placed_edges: set[tuple[int, int, int]],
        block_region=None,
    ):
        """
        Fill a triangular face with glass panes on the inner surface.
        This is only used if curved faces are disabled, as curved faces are
        created using an entire ellipsoid shell.
        """
        # Get the three inner vertices of the face (inset from outer surface)
        v0 = self.inner_vertices[face_vertices[0]]
        v1 = self.inner_vertices[face_vertices[1]]
        v2 = self.inner_vertices[face_vertices[2]]

        # Calculate edge lengths to determine sampling density
        edge1_length = np.linalg.norm(v1 - v0)
        edge2_length = np.linalg.norm(v2 - v1)
        edge3_length = np.linalg.norm(v0 - v2)
        max_edge = max(edge1_length, edge2_length, edge3_length)

        # Sample densely enough to fill gaps
        samples_per_edge = max(5, int(max_edge * 2))

        # Track placed blocks positions
        placed_blocks = set()

        for i in range(samples_per_edge + 1):
            for j in range(samples_per_edge + 1 - i):
                # Barycentric coordinates
                u = (i + 0.3) / samples_per_edge if i < samples_per_edge else i / samples_per_edge
                v = (j + 0.3) / samples_per_edge if j < samples_per_edge else j / samples_per_edge
                w = 1 - u - v

                if w < -0.01:  # Small tolerance
                    continue

                # Calculate point using barycentric coordinates on inner surface
                point = u * v0 + v * v1 + w * v2

                # Convert to block coordinates
                xyz = (
                    int(np.floor(point[0] + self.centre[0])),
                    int(np.floor(point[1] + self.centre[1])),
                    int(np.floor(point[2] + self.centre[2])),
                )

                # Only place if not already an edge block and not already placed face blocks here
                if (
                    xyz not in placed_edges
                    and xyz not in placed_blocks
                    and self._should_place_block(xyz, block_region)
                ):
                    packer.setblock(xyz, self.face_block)
                    placed_blocks.add(xyz)

    def _get_unique_edges(self):
        """Extract unique edges from the face list."""
        edges = set()
        for face in self.dome.f:
            for i in range(3):
                v1_idx = face[i]
                v2_idx = face[(i + 1) % 3]
                # Store edge as sorted tuple to avoid duplicates
                edge = tuple(sorted([v1_idx, v2_idx]))
                edges.add(edge)
        return edges

    def build(self):
        """Build the geodesic dome in Minecraft."""
        ms.echo(
            f"Building geodesic dome with {len(self.dome.v)} vertices and {len(self.dome.f)} faces..."
        )

        # Pre-load block region if only_in_air is enabled
        block_region = None
        if self.only_in_air:
            # Calculate bounding box of the entire dome
            all_points = np.vstack([self.vertices, self.inner_vertices])
            x0 = int(np.floor(all_points[:, 0].min() + self.centre[0])) - 2
            x1 = int(np.ceil(all_points[:, 0].max() + self.centre[0])) + 2
            y0 = int(np.floor(all_points[:, 1].min() + self.centre[1])) - 2
            y1 = int(np.ceil(all_points[:, 1].max() + self.centre[1])) + 2
            z0 = int(np.floor(all_points[:, 2].min() + self.centre[2])) - 2
            z1 = int(np.ceil(all_points[:, 2].max() + self.centre[2])) + 2

            ms.echo(f"Pre-loading block region ({x1 - x0}×{y1 - y0}×{z1 - z0} blocks)...")
            block_region = ms.get_block_region((x0, y0, z0), (x1, y1, z1))

        # Create a BlockPacker for efficient block placement
        packer = ms.BlockPacker()

        # First pass: Create glass shell/faces (if enabled)
        # Faces go first so edges can replace it
        if self.do_faces:
            if self.curved_faces:
                ms.echo("Creating smooth curved _create_smooth_ellipsoid_shell shell...")
                self._create_ellipsoid_shell(packer, block_region)
            else:
                ms.echo("Filling faces with flat triangular panels...")
                for i, face in enumerate(self.dome.f):
                    self._fill_face(face, packer, set(), block_region)

                    if (i + 1) % 100 == 0:
                        ms.echo(f"  Faces: {i + 1}/{len(self.dome.f)}")

                ms.echo(f"Completed {len(self.dome.f)} faces")
        else:
            ms.echo("Face filling disabled - edges only")

        # Second pass: Draw all geodesic edges
        if self.do_edges:
            ms.echo("Drawing geodesic edges...")
            edges = self._get_unique_edges()
            edge_count = 0

            for v1_idx, v2_idx in edges:
                v1 = self.vertices[v1_idx]
                v2 = self.vertices[v2_idx]
                self._draw_edge_with_tracking(v1, v2, packer, block_region)
                edge_count += 1

                if edge_count % 100 == 0:
                    ms.echo(f"  Edges: {edge_count}/{len(edges)}")

            ms.echo(f"Completed {edge_count} edges")
        else:
            ms.echo("Edge drawing disabled")

        # Optional: Plot vertices with marker blocks
        if self.do_vertices:
            ms.echo("Plotting edge vertices...")
            edges = self._get_unique_edges()
            vertex_indices = set()
            for v1_idx, v2_idx in edges:
                vertex_indices.add(v1_idx)
                vertex_indices.add(v2_idx)

            for v_idx in vertex_indices:
                vertex = self.vertices[v_idx]

                # Convert to world coordinates
                xyz = (
                    int(np.floor(vertex[0] + self.centre[0])),
                    int(np.floor(vertex[1] + self.centre[1])),
                    int(np.floor(vertex[2] + self.centre[2])),
                )

                if self._should_place_block(xyz, block_region):
                    packer.setblock(xyz, self.vertex_marker_block)

            ms.echo(f"Plotted {len(vertex_indices)} vertices")

        # Pack and place all blocks
        ms.echo("Placing blocks in world...")

        # Check if packer has any blocks before trying to pack
        if hasattr(packer, "offset") and packer.offset is None:
            ms.echo("Warning: No blocks were added to the packer.")
            return

        blockpack = packer.pack()
        blockpack.write_world()

        ((x0, y0, z0), (x1, y1, z1)) = blockpack.block_bounds()

        ms.await_loaded_region(x0 - 1, z0 - 1, x1 + 1, z1 + 1)

        ms.echo("Geodesic dome complete!")
        print(f"Region: //pos {x0},{y0},{z0} {x1},{y1},{z1}", file=sys.stderr)


if __name__ == "__main__":
    # Get player position as centre
    # Minecraft blocks: centre at (x+0.5, y+1, z+0.5) for proper symmetry
    player_pos = ms.player_position()
    centre_X = int(player_pos[0]) + 0.5
    centre_Y = int(player_pos[1]) + 1
    centre_Z = int(player_pos[2]) + 0.5

    # Create and build the dome using configuration from top of file
    dome = MinecraftGeodesicDome(
        centre_X,
        centre_Y,
        centre_Z,
        radius=RADIUS,
        subdivisions=SUBDIVISIONS,
        edge_thickness_radial=EDGE_THICKNESS_RADIAL,
        edge_thickness_tangential=EDGE_THICKNESS_TANGENTIAL,
        cutoff_layers=CUTOFF_LAYERS,
        edge_block=EDGE_BLOCK,
        face_block=FACE_BLOCK,
        vertex_marker_block=VERTEX_MARKER_BLOCK,
        scale_x=SCALE_X,
        scale_y=SCALE_Y,
        scale_z=SCALE_Z,
        hemisphere_only=HEMISPHERE_ONLY,
        do_edges=DO_EDGES,
        do_vertices=DO_VERTICES,
        do_faces=DO_FACES,
        only_in_air=ONLY_IN_AIR,
        face_inset=FACE_INSET,
        curved_faces=CURVED_FACES,
    )

    dome.build()
