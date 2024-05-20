import * as THREE from 'three';

// Generic method for drawing a wireframe box with lines, including controls for size, position, line thickness, and color
function wireframeBox(size, position, color, linewidth) {
    size = typeof size === 'number' ? new THREE.Vector3(size, size, size) : size;
    const geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
    const material = new THREE.LineBasicMaterial({ color: color, linewidth: linewidth });

    const wireframe = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), material);
    wireframe.position.set(position.x, position.y, position.z);

    return wireframe;
}

// coordinate systems
class CoordinateSystem {
    constructor(scene) {
        this.scene = scene;
        this.origin = new CoordinateOrigin(scene);
        this.object3d = new THREE.Object3D();
        this.scene.add(this.object3d);
        // place the origin in this coordinate system's 3d object
        this.object3d.add(this.origin.object3d);
    }

    update(coordSysData) {
        const basis = coordSysData.basis;
        const origin = basis.origin;
        const quaternion = quaternion_xyzw(basis.orientation);
        this.object3d.position.set(origin[0], origin[1], origin[2]);
        this.object3d.quaternion.set(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
    }
}

// coordinate system origins
class CoordinateOrigin {
    constructor(scene) {
        this.scene = scene;
        this.object3d = new THREE.Object3D();

        const length = 1;
        const arrowHelperX = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), length, 0xff0000);
        const arrowHelperY = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), length, 0x00ff00);
        const arrowHelperZ = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), length, 0x0000ff);

        this.object3d.add(arrowHelperX, arrowHelperY, arrowHelperZ);
        this.scene.add(this.object3d);
    }
}

// collections of points displayed as spheres
class PointCollection {
    constructor(scene) {
        this.scene = scene;
        this.object3d = new THREE.Object3D();
        this.scene.add(this.object3d);
        this.populated = false;
    }

    addPoint(position, color, size, opacity) {
        const geometry = new THREE.SphereGeometry(size, 32, 32);
        const material = new THREE.MeshBasicMaterial({ color: color, transparent: opacity < 1, opacity: opacity });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(position.x, position.y, position.z);
        this.object3d.add(sphere);
        this.populated = true;
    }

    addPoints(x, y, z, color, size, opacity) {
        for (let i = 0; i < x.length; i++) {
            this.addPoint(new THREE.Vector3(x[i], y[i], z[i]), colorToHex(color), size, opacity);
        }
    }

    update(data) {
        if (this.populated == false) {
            this.addPoints(data.x, data.y, data.z, data.color, data.marker_size, data.opacity);
        }
    }
}

// collections of connected line segments
class Line {
    constructor(scene) {
        this.scene = scene;
        this.object3d = new THREE.Object3D();
        this.scene.add(this.object3d);
        this.populated = false;
    }

    addLine(start, end, color, linewidth, opacity) {
        const geometry = new THREE.BufferGeometry();
        const positions = [start.x, start.y, start.z, end.x, end.y, end.z];
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        const material = new THREE.LineBasicMaterial({ color: color, linewidth: linewidth, transparent: opacity < 1, opacity: opacity });
        const line = new THREE.Line(geometry, material);
        this.object3d.add(line);
        this.populated = true;
    }

    addLines(xs, ys, zs, color, linewidth, opacity) {
        for (let i = 0; i < xs.length - 1; i++) {
            this.addLine(new THREE.Vector3(xs[i], ys[i], zs[i]), new THREE.Vector3(xs[i + 1], ys[i + 1], zs[i + 1]), colorToHex(color), linewidth, opacity);
        }
    }

    update(data) {
        if (this.populated == false) {
            this.addLines(data.x, data.y, data.z, data.color, data.linewidth, data.opacity);
        }
    }
}

// wireframe cubes
class Cube {
    constructor(scene) {
        this.scene = scene;
        this.cube = null;
    }
    
    addCube(size, position, orientation, color, linewidth, opacity) {
        console.log('size: ' + size);
        console.log('position: ' + position);
        console.log('orientation: ' + orientation);
        console.log('color: ' + color.getHexString());
        console.log('linewidth: ' + linewidth);
        console.log('opacity: ' + opacity);

        const geometry = new THREE.BoxGeometry(size, size, size);
        const material = new THREE.LineBasicMaterial({ color: color, linewidth: linewidth, transparent: opacity < 1, opacity: opacity });
        const wireframe = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), material);
        wireframe.position.set(position[0], position[1], position[2]);
        const quaternion = quaternion_xyzw(orientation);
        wireframe.quaternion.set(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
        this.scene.add(wireframe);
        this.cube = wireframe;
    }

    update(data) {
        if (this.cube == null) {
            this.addCube(data.size, data.position, data.orientation, colorToHex(data.color), data.line_width, data.opacity);
        }
    }
}

// covariance ellipsoids
class UncertaintyEllipsoid {
    constructor(scene) {
        this.scene = scene;
        this.ellipsoid = null;
    }

    addEllipsoid(position, eigenvalues, eigenvectors, color, linewidth, opacity) { 

        const geometry = new THREE.SphereGeometry(1, 12, 12); // Sphere radius 1
        const edges = new THREE.EdgesGeometry(geometry); // Gets edges of the sphere for line drawing
    
        // Create a material
        const material = new THREE.LineBasicMaterial({
            color: color,
            linewidth: linewidth,
            transparent: opacity < 1,
            opacity: opacity
        });
    
        // Create line segments based on edges geometry
        const ellipsoid = new THREE.LineSegments(edges, material);
    
        ellipsoid.matrixAutoUpdate = false; // We will be updating the matrix manually
    
        // Scale and position matrix
        const scaleMatrix = new THREE.Matrix4().makeScale(
            Math.sqrt(eigenvalues[0]),
            Math.sqrt(eigenvalues[1]),
            Math.sqrt(eigenvalues[2])
        );
    
        const rotationMatrix = new THREE.Matrix4().makeBasis(
            new THREE.Vector3(eigenvectors[0][0], eigenvectors[1][0], eigenvectors[2][0]),
            new THREE.Vector3(eigenvectors[0][1], eigenvectors[1][1], eigenvectors[2][1]),
            new THREE.Vector3(eigenvectors[0][2], eigenvectors[1][2], eigenvectors[2][2])
        );
    
        // Combine rotation and scaling
        ellipsoid.matrix.multiplyMatrices(rotationMatrix, scaleMatrix);
    
        // Set position
        ellipsoid.matrix.setPosition(new THREE.Vector3(position[0], position[1], position[2]));
    
        // Add to scene
        this.scene.add(ellipsoid);
    
        return ellipsoid;
    }

    updateEllipsoid(ellipsoid, position, eigenvalues, eigenvectors) {
        // Scale and position matrix
        const scaleMatrix = new THREE.Matrix4().makeScale(
            Math.sqrt(eigenvalues[0]),
            Math.sqrt(eigenvalues[1]),
            Math.sqrt(eigenvalues[2])
        );
    
        const rotationMatrix = new THREE.Matrix4().makeBasis(
            new THREE.Vector3(eigenvectors[0][0], eigenvectors[1][0], eigenvectors[2][0]),
            new THREE.Vector3(eigenvectors[0][1], eigenvectors[1][1], eigenvectors[2][1]),
            new THREE.Vector3(eigenvectors[0][2], eigenvectors[1][2], eigenvectors[2][2])
        );
    
        // Combine rotation and scaling
        ellipsoid.matrix.multiplyMatrices(rotationMatrix, scaleMatrix);
    
        // Set position
        ellipsoid.matrix.setPosition(new THREE.Vector3(position[0], position[1], position[2]));
    }

    update(data) {
        if (this.ellipsoid == null) {
            this.ellipsoid = this.addEllipsoid(data.position, data.eigenvalues, data.eigenvectors, colorToHex(data.color), data.line_width, data.opacity);
        } else {
            this.updateEllipsoid(this.ellipsoid, data.position, data.eigenvalues, data.eigenvectors);
        }
    }
}

// an environment with a grid floor and a 3d wireframe box representing the working volume
class GridEnvironment {
    constructor(scene) {
        var size = 10;
        var divisions = 10;
        var color = 0xffffff;
        var linewidth = 1;
        this.scene = scene;
        this.grid = new THREE.GridHelper(size, divisions, color, color);
        this.scene.add(this.grid);

        this.box = wireframeBox(size, new THREE.Vector3(0, 0, 0), color, linewidth);
        this.scene.add(this.box);
    }
}

function quaternion_xyzw(q) {
    return [q.x, q.y, q.z, q.w];
}

// utility function converting color tuples to hex. colors can be 3-tuples or 4-tuples with numbers ranging from 0 to 1. They can also already be color strings starting with 0x. Returns a color of the type used in three.js
function colorToHex(color) {
    if (typeof color === 'string') {
        return new THREE.Color(color);
    } else if (color.length === 3) {
        return new THREE.Color(color[0], color[1], color[2]);
    } else if (color.length === 4) {
        return new THREE.Color(color[0], color[1], color[2]);
    }
    if (color.length === 3) {
        return new THREE.Color(color[0], color[1], color[2]);
    } else if (color.length === 4) {
        return new THREE.Color(color[0], color[1], color[2]);
    }
}

export { CoordinateSystem, PointCollection, Line, Cube, UncertaintyEllipsoid, GridEnvironment };