// const genBtn = document.getElementById("genBtn");
// const retBtn = document.getElementById("retBtn");

// const promptEl = document.getElementById("prompt");
// const genStatus = document.getElementById("genStatus");
// const genResult = document.getElementById("genResult");

// const uploadEl = document.getElementById("upload");
// const retStatus = document.getElementById("retStatus");
// const retGrid = document.getElementById("retGrid");

// genBtn.addEventListener("click", async () => {
//   const prompt = promptEl.value.trim();
//   if (!prompt) {
//     genStatus.textContent = "Please enter a prompt.";
//     return;
//   }

//   genStatus.textContent = "Generating...";
//   genResult.innerHTML = "";

//   try {
//     const res = await fetch("/generate", {
//       method: "POST",
//       headers: {"Content-Type": "application/json"},
//       body: JSON.stringify({ prompt })
//     });
//     const data = await res.json();
//     if (data.error) throw new Error(data.error);

    // const img = new Image();
    // img.src = `data:image/png;base64,${data.image_b64}`;
    // img.alt = "Generated image";
    // genResult.innerHTML = "";
    // genResult.appendChild(img);
    // genStatus.textContent = "Done.";
//   } catch (err) {
//     console.error(err);
//     genStatus.textContent = "Generation failed.";
//   }
// });

// retBtn.addEventListener("click", async () => {
//   const file = uploadEl.files?.[0];
//   if (!file) {
//     retStatus.textContent = "Please choose an image to search by.";
//     return;
//   }

//   retStatus.textContent = "Searching similar images...";
//   retGrid.innerHTML = "";

//   const form = new FormData();
//   form.append("image", file);

//   try {
//     const res = await fetch("/retrieve", {
//       method: "POST",
//       body: form
//     });
//     const data = await res.json();

//     if (data.error) throw new Error(data.error);

    // if (data.warn) {
    //   retStatus.textContent = data.warn;
    // } else {
    //   retStatus.textContent = `Found ${data.results.length} results.`;
    // }

    // if (!data.results || data.results.length === 0) {
    //   retGrid.innerHTML = "<p style='color:#aab2d5'>No results. Add images to <code>static/gallery/</code> and restart the server.</p>";
    //   return;
    // }

    // const frag = document.createDocumentFragment();
    // data.results.forEach((url) => {
    //   const img = new Image();
    //   img.src = url;
    //   img.alt = "Retrieved";
    //   frag.appendChild(img);
    // });
    // retGrid.appendChild(frag);
  // } catch (err) {
  //   console.error(err);
  //   retStatus.textContent = "Retrieval failed.";
  // }


// Global variables
let currentPage = 'home';
let scene, camera, renderer, particles;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    // Remove this line
    // initializeTheme();
    initializeTabs();
    initializeImageUpload();
    initializeContactForm();
    
    // Load initial page
    const hash = window.location.hash.slice(1) || 'home';
    navigateTo(hash);
    
    // Generate button
    const generateBtn = document.getElementById('generate-btn');
    if (generateBtn) {
        generateBtn.addEventListener('click', generateImage);
    }
    
    // Find similar button
    const findSimilarBtn = document.getElementById('find-similar-btn');
    if (findSimilarBtn) {
        findSimilarBtn.addEventListener('click', findSimilarImages);
    }

    // Mobile menu toggle
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    
    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
        });

        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!mobileMenuButton.contains(e.target) && !mobileMenu.contains(e.target)) {
                mobileMenu.classList.add('hidden');
            }
        });

        // Close menu when clicking a menu item
        mobileMenu.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                mobileMenu.classList.add('hidden');
            });
        });
    }

    // Set initial page state
    const initialPage = window.location.hash.slice(1) || 'home';
    const initialPageEl = document.getElementById(`${initialPage}-page`);
    
    if (initialPageEl) {
        initialPageEl.classList.remove('hidden');
        requestAnimationFrame(() => {
            initialPageEl.classList.add('active');
        });
        currentPage = initialPage;
        updateActiveNavLink(initialPage);
    }
});

// Navigation functions
function initializeNavigation() {
    // Handle navigation clicks
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const href = e.target.closest('.nav-link').getAttribute('href');
            
            // Check if it's the home link
            if (href === '#home') {
                // Update the path to root URL
                window.location.href = '/';
                return;
            }
            
            // Handle other navigation
            if (href.startsWith('#')) {
                const page = href.slice(1);
                navigateTo(page);
                
                // Close mobile menu if open
                const mobileMenu = document.getElementById('mobile-menu');
                if (mobileMenu) {
                    mobileMenu.classList.add('hidden');
                }
            }
        });
    });

    // Handle mobile menu toggle
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    
    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
        });
    }
}

async function navigateTo(page) {
    // Don't navigate if we're already on this page
    if (currentPage === page) return;

    // Get current and target page elements
    const currentPageEl = document.getElementById(`${currentPage}-page`);
    const targetPageEl = document.getElementById(`${page}-page`);
    
    if (!targetPageEl) return;

    // Remove hidden class from target but keep it invisible with CSS
    targetPageEl.classList.remove('hidden');
    
    // Start transition
    if (currentPageEl) {
        // Fade out current page
        currentPageEl.classList.add('fade-out');
        
        // Wait for animation
        await new Promise(resolve => setTimeout(resolve, 300));
        
        // Hide old page
        currentPageEl.classList.add('hidden');
        currentPageEl.classList.remove('active', 'fade-out');
    }

    // Show new page with animation
    requestAnimationFrame(() => {
        targetPageEl.classList.add('active');
    });

    // Update state
    currentPage = page;
    
    // Update URL
    window.history.pushState(null, '', `#${page}`);
    
    // Update active nav link
    updateActiveNavLink(page);
    
    // Initialize page-specific functionality
    if (page === 'contact') {
        initializeParticles();
    }

    // Scroll to top smoothly
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

function updateActiveNavLink(page) {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('text-cyan-400', 'font-semibold');
        if (link.getAttribute('href') === `#${page}`) {
            link.classList.add('text-cyan-400', 'font-semibold');
        }
    });
}

// Workspace functions
function initializeTabs() {
    const textToImageTab = document.getElementById('tab-text-to-image');
    const imageToImageTab = document.getElementById('tab-image-to-image');
    
    textToImageTab.addEventListener('click', () => showTab('text-to-image'));
    imageToImageTab.addEventListener('click', () => showTab('image-to-image'));
    
    // Initialize text to image functionality
    const generateBtn = document.getElementById('generate-btn');
    generateBtn.addEventListener('click', generateImage);
    
    // Initialize image to image functionality
    const findSimilarBtn = document.getElementById('find-similar-btn');
    findSimilarBtn.addEventListener('click', findSimilarImages);
}

// Update the showTab function to use cyan theme
function showTab(tab) {
    // Update tab buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active', 'text-cyan-400', 'border-cyan-500');
        btn.classList.add('text-gray-400', 'hover:text-cyan-300');
    });
    
    const activeTab = document.getElementById(`tab-${tab}`);
    activeTab.classList.add('active', 'text-cyan-400', 'border-b-2', 'border-cyan-500');
    activeTab.classList.remove('text-gray-400', 'hover:text-cyan-300');
    
    // Update sections
    document.querySelectorAll('.workspace-section').forEach(section => {
        section.classList.add('hidden');
    });
    
    document.getElementById(`${tab}-section`).classList.remove('hidden');
}

// Text to Image functionality
async function generateImage() {
    const prompt = document.getElementById('text-prompt').value.trim();
    const loadingDiv = document.getElementById('text-loading');
    const containerDiv = document.getElementById('generated-image-container');
    const generateBtn = document.getElementById('generate-btn');
    const generatedImage = document.getElementById('generated-image');
    
    if (!prompt) {
        alert('Please enter a prompt for image generation.');
        return;
    }
    
    // Show loading state with more detailed message
    loadingDiv.classList.remove('hidden');
    loadingDiv.innerHTML = `
        <div class="flex flex-col items-center">
            <i class="fas fa-spinner fa-spin text-3xl mb-2 text-cyan-400"></i>
            <p class="text-cyan-400">Generating your image...</p>
            <p class="text-sm text-cyan-300/70 mt-2">This may take 15-30 seconds</p>
        </div>
    `;
    containerDiv.classList.add('hidden');
    generateBtn.disabled = true;
    generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Generating...';
    
    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Generation failed');
        }
        
        const data = await response.json();
        
        // Display the generated image
        generatedImage.src = `data:image/png;base64,${data.image_b64}`;
        generatedImage.alt = `Generated image: ${prompt}`;
        containerDiv.classList.remove('hidden');
        
        // Add download button
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'glass-button px-4 py-2 text-cyan-400 rounded hover:text-cyan-300 transition-all duration-300';
        downloadBtn.innerHTML = '<i class="fas fa-download mr-2"></i>Download Image';
        downloadBtn.onclick = () => {
            const link = document.createElement('a');
            link.href = generatedImage.src;
            link.download = `sdxl-${prompt.slice(0, 30)}.png`;
            link.click();
        };
        containerDiv.appendChild(downloadBtn);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to generate image: ' + error.message);
    } finally {
        loadingDiv.classList.add('hidden');
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<i class="fas fa-magic mr-2"></i>Generate Image';
    }
}

async function simulateImageGeneration() {
    // Simulate API delay
    return new Promise(resolve => {
        setTimeout(resolve, 3000);
    });
}

// Image to Image functionality
function initializeImageUpload() {
    const fileInput = document.getElementById('image-upload');
    const uploadLabel = fileInput.nextElementSibling;
    const previewDiv = document.getElementById('uploaded-image-preview');
    const uploadedImage = document.getElementById('uploaded-image');
    const findBtn = document.getElementById('find-similar-btn');
    
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        
        if (file) {
            if (file.size > 10 * 1024 * 1024) { // 10MB limit
                alert('File size should be less than 10MB.');
                return;
            }
            
            if (!file.type.startsWith('image/')) {
                alert('Please upload a valid image file.');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = (e) => {
                uploadedImage.src = e.target.result;
                previewDiv.classList.remove('hidden');
                findBtn.disabled = false;
                findBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Drag and drop functionality
    const dropZone = uploadLabel.parentElement;
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-cyan-500', 'bg-cyan-900/20');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-cyan-500', 'bg-cyan-900/20');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-cyan-500', 'bg-cyan-900/20');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change'));
        }
    });
}

async function findSimilarImages() {
    const fileInput = document.getElementById('image-upload');
    const loadingDiv = document.getElementById('image-loading');
    const containerDiv = document.getElementById('similar-images-container');
    const findBtn = document.getElementById('find-similar-btn');
    const gridDiv = document.getElementById('similar-images-grid');
    
    if (!fileInput.files || !fileInput.files[0]) {
        alert('Please select an image first.');
        return;
    }
    
    // Show loading state
    loadingDiv.classList.remove('hidden');
    containerDiv.classList.add('hidden');
    findBtn.disabled = true;
    findBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Searching...';
    
    try {
        const formData = new FormData();
        formData.append('image', fileInput.files[0]);
        
        const response = await fetch('/retrieve', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Retrieval failed');
        }
        
        const data = await response.json();
        
        if (data.warn) {
            alert(data.warn);
            return;
        }
        
        // Clear previous results
        gridDiv.innerHTML = '';
        
        // Display similar images
        if (data.results && data.results.length > 0) {
            data.results.forEach(imagePath => {
                const div = document.createElement('div');
                div.className = 'bg-gray-100 dark:bg-gray-700 p-4 rounded-lg';
                
                const img = new Image();
                img.src = imagePath;
                img.alt = 'Similar image';
                img.className = 'w-full h-auto rounded-lg';
                
                div.appendChild(img);
                gridDiv.appendChild(div);
            });
            containerDiv.classList.remove('hidden');
        } else {
            gridDiv.innerHTML = '<p class="text-center text-gray-500">No similar images found</p>';
            containerDiv.classList.remove('hidden');
        }
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to find similar images. Please try again.');
    } finally {
        loadingDiv.classList.add('hidden');
        findBtn.disabled = false;
        findBtn.innerHTML = '<i class="fas fa-search mr-2"></i>Find Similar Images';
    }
}

async function simulateImageSearch() {
    // Simulate API delay
    return new Promise(resolve => {
        setTimeout(resolve, 2500);
    });
}

// Contact form and 3D particles
function initializeContactForm() {
    const form = document.getElementById('contact-form');
    
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        
        // Get form data
        const formData = {
            firstName: document.getElementById('first-name').value,
            lastName: document.getElementById('last-name').value,
            email: document.getElementById('email').value,
            subject: document.getElementById('subject').value,
            message: document.getElementById('message').value
        };
        
        // Simulate form submission
        alert('Thank you for your message! We\'ll get back to you soon.');
        form.reset();
    });
}

function initializeParticles() {
    const canvas = document.getElementById('particle-canvas');
    if (!canvas || scene) return; // Prevent multiple initializations
    
    // Set up Three.js scene
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true });
    
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    
    // Create particles
    const particleCount = 100;
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const velocities = [];
    
    for (let i = 0; i < particleCount; i++) {
        positions[i * 3] = (Math.random() - 0.5) * 20;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 20;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 20;
        
        colors[i * 3] = Math.random() * 0.5 + 0.5;
        colors[i * 3 + 1] = Math.random() * 0.5 + 0.5;
        colors[i * 3 + 2] = 1;
        
        velocities.push({
            x: (Math.random() - 0.5) * 0.02,
            y: (Math.random() - 0.5) * 0.02,
            z: (Math.random() - 0.5) * 0.02
        });
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({
        size: 0.1,
        vertexColors: true,
        transparent: true,
        opacity: 0.8
    });
    
    particles = new THREE.Points(geometry, material);
    scene.add(particles);
    
    camera.position.z = 10;
    
    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        
        if (currentPage === 'contact' && particles) {
            const positions = particles.geometry.attributes.position.array;
            
            for (let i = 0; i < particleCount; i++) {
                positions[i * 3] += velocities[i].x;
                positions[i * 3 + 1] += velocities[i].y;
                positions[i * 3 + 2] += velocities[i].z;
                
                // Bounce off boundaries
                if (positions[i * 3] > 10 || positions[i * 3] < -10) {
                    velocities[i].x *= -1;
                }
                if (positions[i * 3 + 1] > 10 || positions[i * 3 + 1] < -10) {
                    velocities[i].y *= -1;
                }
                if (positions[i * 3 + 2] > 10 || positions[i * 3 + 2] < -10) {
                    velocities[i].z *= -1;
                }
            }
            
            particles.geometry.attributes.position.needsUpdate = true;
            particles.rotation.y += 0.005;
            
            renderer.render(scene, camera);
        }
    }
    
    animate();
}

// Handle browser back/forward
function handlePopState() {
    const page = window.location.hash.slice(1) || 'home';
    navigateTo(page);
}

window.removeEventListener('popstate', handlePopState); // Remove any existing listener
window.addEventListener('popstate', handlePopState);

// Handle window resize
window.addEventListener('resize', () => {
    if (renderer && camera) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
});

// Update the navigation event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Get all pages and nav links
    const pages = document.querySelectorAll('.page');
    const navLinks = document.querySelectorAll('.nav-link');

    // Function to show a specific page and hide others
    const showPage = (pageId) => {
        pages.forEach(page => {
            page.classList.add('hidden');
        });
        const pageToShow = document.getElementById(`${pageId}-page`);
        if (pageToShow) {
            pageToShow.classList.remove('hidden');
            // Update active state of nav links
            navLinks.forEach(link => {
                const linkPage = link.getAttribute('href').substring(1);
                if (linkPage === pageId) {
                    link.classList.add('active', 'text-cyan-400');
                    link.classList.remove('text-gray-400');
                } else {
                    link.classList.remove('active', 'text-cyan-400');
                    link.classList.add('text-gray-400');
                }
            });
        }
    };

    // Handle navigation clicks
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const pageId = e.target.closest('.nav-link').getAttribute('href').substring(1);
            window.location.hash = pageId;
            showPage(pageId);
        });
    });

    // Handle hash changes
    window.addEventListener('hashchange', () => {
        const hash = window.location.hash.substring(1) || 'home';
        showPage(hash);
    });

    // Handle initial load
    const initialHash = window.location.hash.substring(1) || 'home';
    showPage(initialHash);
});

// Add event listeners for suggested prompts
document.addEventListener('DOMContentLoaded', function() {
    const textPrompt = document.getElementById('text-prompt');
    const suggestedPrompts = document.querySelectorAll('.suggested-prompt');

    suggestedPrompts.forEach(button => {
        button.addEventListener('click', function() {
            textPrompt.value = this.textContent.trim();
            // Optional: scroll the textarea into view
            textPrompt.scrollIntoView({ behavior: 'smooth' });
            // Optional: add focus to the textarea
            textPrompt.focus();
        });
    });
});