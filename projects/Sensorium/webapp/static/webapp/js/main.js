/**
 * Main JavaScript for Sensorium Dashboard
 * Handles dynamic updates, AJAX calls, and grid generation
 */

let currentMouseId = null;
let currentVideoId = null;
let currentNeuronId = null;

/**
 * Initialize the app with default values
 */
function initializeApp(mouseId, videoId, neuronId) {
    currentMouseId = mouseId;
    currentVideoId = videoId;
    currentNeuronId = neuronId;
    
    // Load videos dropdown
    loadVideos(mouseId, videoId);
    
    // Load 3D plot
    load3DPlot(mouseId);
    
    // Load video player and metadata
    if (videoId) {
        loadVideoPlayer(mouseId, videoId);
        loadMetadata(mouseId, videoId);
        loadNeurons(mouseId, videoId, neuronId);
        loadAllGrids(mouseId, videoId, neuronId);
    }
}

/**
 * Load videos for a mouse
 */
function loadVideos(mouseId, selectedVideoId = null) {
    fetch(`/api/mice/${mouseId}/videos/`)
        .then(response => response.json())
        .then(data => {
            const videoSelect = document.getElementById('video-select');
            videoSelect.innerHTML = '<option value="">-- Select a video --</option>';
            
            if (data.videos && data.videos.length > 0) {
                data.videos.forEach(videoId => {
                    const option = document.createElement('option');
                    option.value = videoId;
                    option.textContent = videoId;
                    if (selectedVideoId && videoId === selectedVideoId) {
                        option.selected = true;
                    }
                    videoSelect.appendChild(option);
                });
                videoSelect.disabled = false;
            }
        })
        .catch(error => {
            console.error('Error loading videos:', error);
            showError('video-select', 'Failed to load videos');
        });
}

/**
 * Load 3D cell motor coordinates plot
 */
function load3DPlot(mouseId) {
    const plotContainer = document.getElementById('cell-motor-3d-plot');
    plotContainer.innerHTML = '<div class="loading">Loading 3D plot...</div>';
    
    fetch(`/api/mice/${mouseId}/cell_coordinates/`)
        .then(response => response.json())
        .then(data => {
            if (data.coordinates && data.coordinates.length > 0) {
                const fig = plotCellCoordinates3D(data.coordinates);
                // Ensure container has proper dimensions
                plotContainer.style.width = '100%';
                plotContainer.style.height = '500px';
                plotContainer.style.minHeight = '500px';
                plotContainer.style.margin = '0';
                plotContainer.style.padding = '0';
                plotContainer.innerHTML = ''; // Clear loading message
                
                // Update layout to ensure proper sizing
                fig.layout.width = null; // Let it fill container
                fig.layout.height = 500;
                
                Plotly.newPlot('cell-motor-3d-plot', fig.data, fig.layout, {
                    responsive: true,
                    displayModeBar: true,
                    staticPlot: false
                }).then(() => {
                    // Force resize to ensure proper centering
                    setTimeout(() => {
                        Plotly.Plots.resize('cell-motor-3d-plot');
                    }, 100);
                });
            } else {
                plotContainer.innerHTML = '<div class="error">No cell coordinates available</div>';
            }
        })
        .catch(error => {
            console.error('Error loading 3D plot:', error);
            plotContainer.innerHTML = '<div class="error">Failed to load 3D plot</div>';
        });
}

/**
 * Load video player
 */
function loadVideoPlayer(mouseId, videoId) {
    const container = document.getElementById('video-player-container');
    const labelContainer = document.getElementById('video-label-container');
    container.innerHTML = '<div class="loading">Loading video...</div>';
    labelContainer.innerHTML = ''; // Clear label
    
    // Fetch video info to get label
    const videoInfoPromise = fetch(`/api/mice/${mouseId}/videos/${videoId}/info/`)
        .then(response => {
            if (!response.ok) {
                return null; // If info fetch fails, continue without label
            }
            return response.json();
        })
        .catch(error => {
            console.warn('Failed to fetch video info:', error);
            return null;
        });
    
    // Fetch video data
    const videoDataPromise = fetch(`/api/mice/${mouseId}/videos/${videoId}/video/`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        });
    
    // Process both promises
    Promise.all([videoInfoPromise, videoDataPromise])
        .then(([videoInfo, data]) => {
            // Display label if available
            if (videoInfo && videoInfo.label) {
                labelContainer.textContent = videoInfo.label;
            } else {
                labelContainer.innerHTML = '';
            }
            
            if (data.error) {
                container.innerHTML = `<div class="error">${data.error}</div>`;
                return;
            }
            if (data.video_data) {
                // Ensure data URI format
                let videoSrc = data.video_data;
                if (!videoSrc.startsWith('data:')) {
                    videoSrc = `data:video/mp4;base64,${videoSrc}`;
                }
                
                // Create video element with proper attributes
                const video = document.createElement('video');
                video.controls = true;
                video.preload = 'auto';
                video.playsInline = true;
                video.style.width = '100%';
                video.style.height = '100%';
                video.style.maxHeight = '100%';
                video.style.borderRadius = '4px';
                video.style.objectFit = 'contain';
                video.style.backgroundColor = '#000';
                
                // Clear container first
                container.innerHTML = '';
                
                // Set source directly on video element (more reliable than source tag)
                video.src = videoSrc;
                
                // Append video to container
                container.appendChild(video);
                
                // Force video to load
                video.load();
                
                // Add event listeners for better error handling and debugging
                video.addEventListener('loadstart', function() {
                    console.log('Video load started');
                });
                
                video.addEventListener('loadeddata', function() {
                    console.log('Video data loaded successfully');
                });
                
                video.addEventListener('loadedmetadata', function() {
                    console.log('Video metadata loaded:', {
                        duration: video.duration,
                        videoWidth: video.videoWidth,
                        videoHeight: video.videoHeight
                    });
                });
                
                video.addEventListener('canplay', function() {
                    console.log('Video can start playing');
                });
                
                video.addEventListener('error', function(e) {
                    console.error('Video loading error:', e);
                    console.error('Video error details:', video.error);
                    console.error('Video src length:', videoSrc.length);
                    console.error('Video src preview:', videoSrc.substring(0, 100) + '...');
                    
                    let errorMsg = 'Video failed to load.';
                    if (video.error) {
                        switch(video.error.code) {
                            case video.error.MEDIA_ERR_ABORTED:
                                errorMsg = 'Video loading aborted.';
                                break;
                            case video.error.MEDIA_ERR_NETWORK:
                                errorMsg = 'Network error while loading video.';
                                break;
                            case video.error.MEDIA_ERR_DECODE:
                                errorMsg = 'Video decoding error. The video codec (mp4v) may not be supported by your browser.';
                                break;
                            case video.error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                                errorMsg = 'Video format not supported by browser. The browser may not support mp4v codec.';
                                break;
                        }
                    }
                    // Don't replace the video element, just show error message
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'error';
                    errorDiv.style.position = 'absolute';
                    errorDiv.style.top = '10px';
                    errorDiv.style.left = '10px';
                    errorDiv.style.zIndex = '10';
                    errorDiv.style.backgroundColor = 'rgba(255, 0, 0, 0.8)';
                    errorDiv.style.color = 'white';
                    errorDiv.style.padding = '10px';
                    errorDiv.style.borderRadius = '4px';
                    errorDiv.innerHTML = `${errorMsg}<br><small>Video element is present but cannot play. Codec issue.</small>`;
                    container.appendChild(errorDiv);
                });
            } else {
                container.innerHTML = '<div class="error">Video not available</div>';
            }
        })
        .catch(error => {
            console.error('Error loading video:', error);
            container.innerHTML = '<div class="error">Failed to load video: ' + error.message + '</div>';
            labelContainer.innerHTML = ''; // Clear label on error
        });
}

/**
 * Load metadata
 */
function loadMetadata(mouseId, videoId) {
    const container = document.getElementById('metadata-box');
    container.innerHTML = '<div class="loading">Loading metadata...</div>';
    
    fetch(`/api/mice/${mouseId}/videos/${videoId}/info/`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                container.innerHTML = `<div class="error">${data.error}</div>`;
                return;
            }
            
            let html = '';
            html += `<p><strong>Video ID:</strong> ${videoId}</p>`;
            
            // Video Valid Frames
            if (data.video_valid_frames !== undefined) {
                html += `<p><strong>Total Valid Frames:</strong> ${data.video_valid_frames}</p>`;
            }
            
            // Number of Equivalent Videos
            if (data.number_equivalent_videos !== undefined) {
                html += `<p><strong>Number of Equivalent Videos:</strong> ${data.number_equivalent_videos}</p>`;
            }
            
            // Consistent Responses Indicator
            if (data.same_valid_responses !== undefined) {
                const consistencyLabel = data.same_valid_responses ? '✓ Yes' : '✗ No';
                html += `<p><strong>Consistent Responses:</strong> ${consistencyLabel}</p>`;
            }
            
            // Incorrect Valid Responses
            if (data.incorrect_valid_responses !== undefined) {
                html += `<p><strong>Incorrect Responses on Valid Frames:</strong> ${data.incorrect_valid_responses}</p>`;
            }
            
            // Equivalent Video IDs
            if (data.equivalent_videos && data.equivalent_videos.length > 0) {
                html += `<p><strong>Equivalent Video IDs:</strong> ${data.equivalent_videos.join(', ')}</p>`;
            }
            
            container.innerHTML = html;
        })
        .catch(error => {
            console.error('Error loading metadata:', error);
            container.innerHTML = '<div class="error">Failed to load metadata</div>';
        });
}

/**
 * Load neurons dropdown
 */
function loadNeurons(mouseId, videoId, selectedNeuronId = null) {
    fetch(`/api/mice/${mouseId}/videos/${videoId}/neurons/`)
        .then(response => response.json())
        .then(data => {
            const neuronSelect = document.getElementById('neuron-select');
            neuronSelect.innerHTML = '<option value="">-- Select a neuron --</option>';
            
            if (data.num_neurons) {
                const numNeurons = Math.min(data.num_neurons, 1000); // Limit dropdown size
                for (let i = 0; i < numNeurons; i++) {
                    const option = document.createElement('option');
                    option.value = i;
                    option.textContent = `Neuron ${i}`;
                    if (selectedNeuronId !== null && i === selectedNeuronId) {
                        option.selected = true;
                    }
                    neuronSelect.appendChild(option);
                }
                neuronSelect.disabled = false;
            }
        })
        .catch(error => {
            console.error('Error loading neurons:', error);
            showError('neuron-select', 'Failed to load neurons');
        });
}

/**
 * Create video column header
 */
function createVideoColumnHeader(videoId, isRepresentative = false) {
    const header = document.createElement('div');
    header.className = 'video-column-header';
    header.textContent = isRepresentative ? `REPRESENTATIVE VIDEO ID: ${videoId}` : `EQUIVALENT VIDEO ID: ${videoId}`;
    return header;
}

/**
 * Create plot cell
 */
function createPlotCell(videoId, plotData, plotType, neuronId = null) {
    const cell = document.createElement('div');
    cell.className = 'plot-cell';
    
    const plotDiv = document.createElement('div');
    plotDiv.className = 'plot-container';
    plotDiv.id = `plot-${plotType}-${videoId}`;
    cell.appendChild(plotDiv);
    
    // Store plot data and type for rendering after DOM insertion
    plotDiv.dataset.plotData = JSON.stringify(plotData);
    plotDiv.dataset.plotType = plotType;
    plotDiv.dataset.neuronId = neuronId !== null ? neuronId : '';
    plotDiv.dataset.videoId = videoId;
    
    return cell;
}

/**
 * Render plot in an element (called after element is in DOM)
 */
function renderPlotInElement(elementId) {
    const element = document.getElementById(elementId);
    if (!element) {
        console.error(`Plot element ${elementId} not found`);
        return;
    }
    
    const plotData = JSON.parse(element.dataset.plotData);
    const plotType = element.dataset.plotType;
    const neuronId = element.dataset.neuronId ? parseInt(element.dataset.neuronId) : null;
    const videoId = element.dataset.videoId;
    
    let fig;
    if (plotType === 'responses') {
        fig = plotSingleNeuronResponse(plotData, neuronId, videoId);
    } else if (plotType === 'behavior') {
        fig = plotSingleBehavior(plotData, videoId);
    } else if (plotType === 'pupil_center') {
        fig = plotSinglePupilCenter(plotData, videoId);
    }
    
    if (fig) {
        Plotly.newPlot(elementId, fig.data, fig.layout, {responsive: true});
    } else {
        element.innerHTML = '<div class="error">Failed to create plot</div>';
    }
}

/**
 * Create grid for a data type
 */
function createDataGrid(videoIds, plotData, plotType, neuronId = null) {
    const grid = document.createElement('div');
    grid.className = 'data-grid';
    
    videoIds.forEach((videoId, index) => {
        const isRepresentative = index === 0;
        
        // Create column container
        const column = document.createElement('div');
        column.className = 'video-column';
        
        // Header
        const header = createVideoColumnHeader(videoId, isRepresentative);
        column.appendChild(header);
        
        // Plot cell
        const plotDataForVideo = plotData[videoId];
        if (plotDataForVideo) {
            const cell = createPlotCell(videoId, plotDataForVideo, plotType, neuronId);
            column.appendChild(cell);
        } else {
            const cell = document.createElement('div');
            cell.className = 'plot-cell';
            cell.innerHTML = '<div class="error">No data available</div>';
            column.appendChild(cell);
        }
        
        // Append column to grid
        grid.appendChild(column);
    });
    
    return grid;
}

/**
 * Load all grids (neuron responses, behavior, pupil center)
 */
function loadAllGrids(mouseId, videoId, neuronId) {
    // Get video info to get all video IDs
    fetch(`/api/mice/${mouseId}/videos/${videoId}/info/`)
        .then(response => response.json())
        .then(videoInfo => {
            const allVideoIds = [videoId];
            if (videoInfo.equivalent_videos) {
                allVideoIds.push(...videoInfo.equivalent_videos);
            }
            
            // Load neuron responses
            loadPlotGrid(mouseId, videoId, allVideoIds, 'responses', neuronId);
            
            // Load behavior
            loadPlotGrid(mouseId, videoId, allVideoIds, 'behavior', null);
            
            // Load pupil center
            loadPlotGrid(mouseId, videoId, allVideoIds, 'pupil_center', null);
        })
        .catch(error => {
            console.error('Error loading video info:', error);
        });
}

/**
 * Load a single plot grid
 */
function loadPlotGrid(mouseId, videoId, videoIds, dataType, neuronId) {
    // Map frontend dataType to backend API data_type
    const apiDataType = dataType === 'responses' ? 'responses' : dataType;
    const containerId = dataType === 'responses' ? 'neuron-responses-container' :
                       dataType === 'behavior' ? 'behavior-container' : 'pupil-center-container';
    const gridId = dataType === 'responses' ? 'neuron-responses-grid' :
                   dataType === 'behavior' ? 'behavior-grid' : 'pupil-center-grid';
    
    const container = document.getElementById(containerId);
    const grid = document.getElementById(gridId);
    
    grid.innerHTML = '<div class="loading">Loading plots...</div>';
    
    fetch(`/api/mice/${mouseId}/videos/${videoId}/plot/${apiDataType}/`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                grid.innerHTML = `<div class="error">${data.error}</div>`;
                return;
            }
            
            // Clear grid
            grid.innerHTML = '';
            
            // Get video IDs and plot data
            const plotData = data.plot_data || {};
            const videoIdsToPlot = data.video_ids || videoIds;
            
            // Create grid columns for each video (each column = header + plot)
            videoIdsToPlot.forEach((vidId, index) => {
                const isRepresentative = index === 0;
                
                // Create column container
                const column = document.createElement('div');
                column.className = 'video-column';
                
                // Create header
                const header = createVideoColumnHeader(vidId, isRepresentative);
                column.appendChild(header);
                
                // Create plot cell
                const plotDataForVideo = plotData[String(vidId)];
                if (plotDataForVideo) {
                    const cell = createPlotCell(vidId, plotDataForVideo, dataType, neuronId);
                    column.appendChild(cell);
                } else {
                    const cell = document.createElement('div');
                    cell.className = 'plot-cell';
                    cell.innerHTML = '<div class="error">No data available</div>';
                    column.appendChild(cell);
                }
                
                // Append column to grid
                grid.appendChild(column);
            });
            
            // Render all plots after DOM is updated
            // Use double requestAnimationFrame to ensure DOM is fully ready
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    videoIdsToPlot.forEach((vidId) => {
                        const plotElementId = `plot-${dataType}-${vidId}`;
                        const plotElement = document.getElementById(plotElementId);
                        if (plotElement && plotElement.dataset.plotData) {
                            try {
                                renderPlotInElement(plotElementId);
                            } catch (error) {
                                console.error(`Error rendering plot ${plotElementId}:`, error);
                                plotElement.innerHTML = '<div class="error">Failed to render plot</div>';
                            }
                        } else if (!plotElement) {
                            console.warn(`Plot element ${plotElementId} not found in DOM`);
                        }
                    });
                });
            });
        })
        .catch(error => {
            console.error(`Error loading ${dataType}:`, error);
            // Don't overwrite grid if plots are already being rendered
            if (grid.innerHTML === '<div class="loading">Loading plots...</div>') {
                grid.innerHTML = '<div class="error">Failed to load plots: ' + error.message + '</div>';
            }
        });
}

/**
 * Show error message
 */
function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<div class="error">${message}</div>`;
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Mouse selection
    const mouseSelect = document.getElementById('mouse-id-select');
    mouseSelect.addEventListener('change', function() {
        const mouseId = this.value;
        if (mouseId) {
            currentMouseId = mouseId;
            currentVideoId = null;
            currentNeuronId = null;
            
            // Reset video and neuron dropdowns
            document.getElementById('video-select').disabled = true;
            document.getElementById('neuron-select').disabled = true;
            
            // Clear video label
            document.getElementById('video-label-container').innerHTML = '';
            
            // Load videos
            loadVideos(mouseId);
            
            // Load 3D plot
            load3DPlot(mouseId);
            
            // Clear grids
            document.getElementById('neuron-responses-grid').innerHTML = '';
            document.getElementById('behavior-grid').innerHTML = '';
            document.getElementById('pupil-center-grid').innerHTML = '';
        }
    });
    
    // Video selection
    const videoSelect = document.getElementById('video-select');
    videoSelect.addEventListener('change', function() {
        const videoId = this.value;
        if (videoId && currentMouseId) {
            currentVideoId = videoId;
            
            // Load video player and metadata
            loadVideoPlayer(currentMouseId, videoId);
            loadMetadata(currentMouseId, videoId);
            
            // Load neurons
            loadNeurons(currentMouseId, videoId, 0);
            
            // Load all grids with default neuron 0
            loadAllGrids(currentMouseId, videoId, 0);
            currentNeuronId = 0;
        }
    });
    
    // Neuron selection
    const neuronSelect = document.getElementById('neuron-select');
    neuronSelect.addEventListener('change', function() {
        const neuronId = parseInt(this.value);
        if (!isNaN(neuronId) && currentMouseId && currentVideoId) {
            currentNeuronId = neuronId;
            
            // Update only neuron responses grid
            fetch(`/api/mice/${currentMouseId}/videos/${currentVideoId}/info/`)
                .then(response => response.json())
                .then(videoInfo => {
                    const allVideoIds = [currentVideoId];
                    if (videoInfo.equivalent_videos) {
                        allVideoIds.push(...videoInfo.equivalent_videos);
                    }
                    loadPlotGrid(currentMouseId, currentVideoId, allVideoIds, 'responses', neuronId);
                })
                .catch(error => {
                    console.error('Error updating neuron plots:', error);
                });
        }
    });
});
