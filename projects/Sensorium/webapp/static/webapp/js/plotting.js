/**
 * Plotting functions using Plotly.js
 * Client-side plotting for Sensorium dashboard
 */

/**
 * Plot single neuron response for one video
 */
function plotSingleNeuronResponse(data, neuronId, videoId) {
    if (!data || data.length === 0) {
        return {
            data: [],
            layout: {
                title: "No neural responses",
                height: 400,
                xaxis: {title: "Frame"},
                yaxis: {title: "Activity"}
            }
        };
    }

    // Handle 2D array: assume (n_neurons, n_frames) or (n_frames, n_neurons)
    let arr = data;
    let n0 = arr.length;
    let n1 = arr[0] ? arr[0].length : 0;
    
    // Determine orientation
    let arrNeuronsByFrames;
    if (n0 > n1) {
        arrNeuronsByFrames = arr; // (n_neurons, n_frames)
    } else {
        // Transpose: (n_frames, n_neurons) -> (n_neurons, n_frames)
        arrNeuronsByFrames = [];
        for (let i = 0; i < n1; i++) {
            arrNeuronsByFrames[i] = [];
            for (let j = 0; j < n0; j++) {
                arrNeuronsByFrames[i][j] = arr[j][i];
            }
        }
    }

    // Extract trace for selected neuron
    let traceData;
    let traceName;
    if (neuronId === null || neuronId === undefined) {
        // Mean across neurons
        traceData = [];
        let nFrames = arrNeuronsByFrames[0].length;
        for (let f = 0; f < nFrames; f++) {
            let sum = 0;
            for (let n = 0; n < arrNeuronsByFrames.length; n++) {
                sum += arrNeuronsByFrames[n][f];
            }
            traceData.push(sum / arrNeuronsByFrames.length);
        }
        traceName = videoId ? `Video ${videoId} (mean)` : "Mean Population";
    } else {
        if (neuronId >= arrNeuronsByFrames.length) {
            return {
                data: [],
                layout: {
                    title: `Neuron ID ${neuronId} out of range`,
                    height: 400
                }
            };
        }
        traceData = arrNeuronsByFrames[neuronId];
        traceName = videoId ? `Video ${videoId} (neuron ${neuronId})` : `Neuron ${neuronId}`;
    }

    // Create x-axis (frame numbers)
    let xAxis = [];
    for (let i = 0; i < traceData.length; i++) {
        xAxis.push(i);
    }

    let plot = {
        x: xAxis,
        y: traceData,
        mode: 'lines',
        name: traceName,
        line: {width: 2},
        hovertemplate: `<b>${traceName}</b><br>Frame: %{x}<br>Activity: %{y:.2f}<br><extra></extra>`
    };

    return {
        data: [plot],
        layout: {
            title: "",
            height: 400,
            xaxis: {title: "Frame", showgrid: true, gridcolor: "#e0e0e0"},
            yaxis: {title: "Activity", showgrid: true, gridcolor: "#e0e0e0"},
            margin: {l: 50, r: 20, t: 10, b: 50},
            plot_bgcolor: "white",
            paper_bgcolor: "white",
            hovermode: "closest"
        }
    };
}

/**
 * Plot single behavior data (running speed + pupil dilation)
 */
function plotSingleBehavior(data, videoId) {
    if (!data || data.length === 0) {
        return {
            data: [],
            layout: {
                title: "No behavior data",
                height: 400,
                xaxis: {title: "Frame"},
                yaxis: {title: "Running Speed", side: "left"},
                yaxis2: {title: "Pupil Dilation", overlaying: "y", side: "right"}
            }
        };
    }

    let arr = data;
    let running, pupil;

    // Determine orientation: (2, n_frames) or (n_frames, 2)
    // Check if first dimension is 2 (meaning 2 rows)
    if (arr.length === 2 && Array.isArray(arr[0]) && Array.isArray(arr[1])) {
        // Shape is (2, n_frames): [running_array, pupil_array]
        running = arr[0];
        pupil = arr[1];
    } else if (arr.length > 2 && Array.isArray(arr[0]) && arr[0].length === 2) {
        // Shape is (n_frames, 2): each element is [running, pupil]
        running = [];
        pupil = [];
        for (let i = 0; i < arr.length; i++) {
            if (Array.isArray(arr[i]) && arr[i].length >= 2) {
                running.push(arr[i][0]);
                pupil.push(arr[i][1]);
            }
        }
    } else {
        // Fallback: assume (2, n_frames) if we can't determine
        console.warn('Unexpected behavior data shape, assuming (2, n_frames)');
        if (arr.length >= 2) {
            running = arr[0] || [];
            pupil = arr[1] || [];
        } else {
            running = [];
            pupil = [];
        }
    }

    // Create x-axis
    let xAxis = [];
    for (let i = 0; i < running.length; i++) {
        xAxis.push(i);
    }

    let runningName = videoId ? `Video ${videoId} - Running` : "Running Speed";
    let pupilName = videoId ? `Video ${videoId} - Pupil` : "Pupil Dilation";

    return {
        data: [
            {
                x: xAxis,
                y: running,
                mode: 'lines',
                name: runningName,
                yaxis: 'y',
                line: {width: 2, color: '#1f77b4'},
                hovertemplate: `<b>${runningName}</b><br>Frame: %{x}<br>Speed: %{y:.2f}<br><extra></extra>`
            },
            {
                x: xAxis,
                y: pupil,
                mode: 'lines',
                name: pupilName,
                yaxis: 'y2',
                line: {width: 2, color: '#ff7f0e'},
                hovertemplate: `<b>${pupilName}</b><br>Frame: %{x}<br>Dilation: %{y:.2f}<br><extra></extra>`
            }
        ],
        layout: {
            title: "",
            height: 400,
            xaxis: {title: "Frame", showgrid: true, gridcolor: "#e0e0e0"},
            yaxis: {title: "Running Speed", side: "left", showgrid: true, gridcolor: "#e0e0e0"},
            yaxis2: {title: "Pupil Dilation", overlaying: "y", side: "right", showgrid: false},
            margin: {l: 50, r: 50, t: 10, b: 50},
            plot_bgcolor: "white",
            paper_bgcolor: "white",
            hovermode: "closest",
            legend: {orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1}
        }
    };
}

/**
 * Plot single pupil center data (horizontal + vertical position)
 */
function plotSinglePupilCenter(data, videoId) {
    if (!data || data.length === 0) {
        return {
            data: [],
            layout: {
                title: "No pupil center data",
                height: 400,
                xaxis: {title: "Frame"},
                yaxis: {title: "Horizontal Position (X)", side: "left"},
                yaxis2: {title: "Vertical Position (Y)", overlaying: "y", side: "right"}
            }
        };
    }

    let arr = data;
    let x, y;

    // Determine orientation: (2, n_frames) or (n_frames, 2)
    // Check if first dimension is 2 (meaning 2 rows)
    if (arr.length === 2 && Array.isArray(arr[0]) && Array.isArray(arr[1])) {
        // Shape is (2, n_frames): [x_array, y_array]
        x = arr[0];
        y = arr[1];
    } else if (arr.length > 2 && Array.isArray(arr[0]) && arr[0].length === 2) {
        // Shape is (n_frames, 2): each element is [x, y]
        x = [];
        y = [];
        for (let i = 0; i < arr.length; i++) {
            if (Array.isArray(arr[i]) && arr[i].length >= 2) {
                x.push(arr[i][0]);
                y.push(arr[i][1]);
            }
        }
    } else {
        // Fallback: assume (2, n_frames) if we can't determine
        console.warn('Unexpected pupil center data shape, assuming (2, n_frames)');
        if (arr.length >= 2) {
            x = arr[0] || [];
            y = arr[1] || [];
        } else {
            x = [];
            y = [];
        }
    }

    // Create x-axis
    let xAxis = [];
    for (let i = 0; i < x.length; i++) {
        xAxis.push(i);
    }

    let xName = videoId ? `Video ${videoId} - X` : "Horizontal (X)";
    let yName = videoId ? `Video ${videoId} - Y` : "Vertical (Y)";

    return {
        data: [
            {
                x: xAxis,
                y: x,
                mode: 'lines',
                name: xName,
                yaxis: 'y',
                line: {width: 2, color: '#2ca02c'},
                hovertemplate: `<b>${xName}</b><br>Frame: %{x}<br>X: %{y:.2f}<br><extra></extra>`
            },
            {
                x: xAxis,
                y: y,
                mode: 'lines',
                name: yName,
                yaxis: 'y2',
                line: {width: 2, color: '#d62728'},
                hovertemplate: `<b>${yName}</b><br>Frame: %{x}<br>Y: %{y:.2f}<br><extra></extra>`
            }
        ],
        layout: {
            title: "",
            height: 400,
            xaxis: {title: "Frame", showgrid: true, gridcolor: "#e0e0e0"},
            yaxis: {title: "Horizontal Position (X)", side: "left", showgrid: true, gridcolor: "#e0e0e0"},
            yaxis2: {title: "Vertical Position (Y)", overlaying: "y", side: "right", showgrid: false},
            margin: {l: 50, r: 50, t: 10, b: 50},
            plot_bgcolor: "white",
            paper_bgcolor: "white",
            hovermode: "closest",
            legend: {orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1}
        }
    };
}

/**
 * Plot 3D cell motor coordinates
 */
function plotCellCoordinates3D(coordinates) {
    if (!coordinates || coordinates.length === 0) {
        return {
            data: [],
            layout: {
                title: "No cell motor coordinates available",
                height: 500
            }
        };
    }

    let x = [];
    let y = [];
    let z = [];

    for (let i = 0; i < coordinates.length; i++) {
        x.push(coordinates[i][0]);
        y.push(coordinates[i][1]);
        z.push(coordinates[i][2]);
    }

    // Calculate center and ranges for proper centering
    let xMin = Math.min(...x);
    let xMax = Math.max(...x);
    let yMin = Math.min(...y);
    let yMax = Math.max(...y);
    let zMin = Math.min(...z);
    let zMax = Math.max(...z);
    
    let xCenter = (xMin + xMax) / 2;
    let yCenter = (yMin + yMax) / 2;
    let zCenter = (zMin + zMax) / 2;
    
    let xRange = xMax - xMin;
    let yRange = yMax - yMin;
    let zRange = zMax - zMin;
    let maxRange = Math.max(xRange, yRange, zRange);

    return {
        data: [{
            type: 'scatter3d',  // Explicitly set 3D scatter plot type
            x: x,
            y: y,
            z: z,
            mode: 'markers',
            marker: {
                size: 3,
                color: z,
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {
                    title: "Z coordinate",
                    len: 0.5,
                    y: 0.5,
                    yanchor: 'middle'
                }
            },
            name: "Neurons",
            hovertemplate: '<b>Neuron</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br><extra></extra>'
        }],
        layout: {
            title: {
                text: "Cell Motor Coordinates (3D)",
                x: 0.5,
                xanchor: 'center'
            },
            scene: {
                xaxis: {
                    title: "X",
                    range: [xCenter - maxRange/2, xCenter + maxRange/2],
                    showbackground: false
                },
                yaxis: {
                    title: "Y",
                    range: [yCenter - maxRange/2, yCenter + maxRange/2],
                    showbackground: false
                },
                zaxis: {
                    title: "Z",
                    range: [zCenter - maxRange/2, zCenter + maxRange/2],
                    showbackground: false
                },
                aspectmode: "cube",  // Use cube for equal aspect ratio
                camera: {
                    eye: {
                        x: 1.5,
                        y: 1.5,
                        z: 1.5
                    },
                    center: {
                        x: 0,
                        y: 0,
                        z: 0
                    },
                    up: {
                        x: 0,
                        y: 0,
                        z: 1
                    }
                },
                bgcolor: "white"
            },
            height: 500,
            margin: {l: 0, r: 0, t: 50, b: 0, autoexpand: false},
            paper_bgcolor: "white",
            autosize: true
        }
    };
}

