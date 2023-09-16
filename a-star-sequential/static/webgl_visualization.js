let scene, camera, renderer, controls;

function init() {
    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 5;

    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('container').appendChild(renderer.domElement);

    // Initialize controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);

    // Add your grid visualization code here
    // (Use Three.js to create geometry, materials, and add objects to the scene)
}

function animate() {
    requestAnimationFrame(animate);
    controls.update(); // Update controls
    renderer.render(scene, camera);
}

init();
animate();
