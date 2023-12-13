let video; //stores webcam video
let posenet; //initialize with ml5 object poseNet
let brain; //initialize with ml5 object neuralNetwork
let fileState = "modelDeployed"; //"modelDeployed" "dataCollection" "modelTraining"
let table; //to store poseToAudio as csv
let rows; //to store reference to poseToAudio csv rows

let pose; //regularly stores the first pose in poses
let skeleton; //regularly stores the first skeleton in poses

let state = "waiting";
let targetLabel;
let poseInput; // to name poses being captured dynamically
let poseLabel = "unknown";
let poseLabelScore = 0.0;

let videoScaleVal = 1;

function preload() {
    table = loadTable("poseToAudio.csv", "csv", "header");
}

function setup() {
    createCanvas(windowWidth, windowWidth);
    noStroke();
    rectMode(CENTER);

    rows = table.getRows();
    video  = createCapture(VIDEO);
    video.hide();

    poseInput = createInput('');
    poseInput.position(1*video.width, video.height*3);
    poseInput.size(400, 32);

    if (fileState != "modelTraining") {initializePosenet();}
    initializeBrain();
}

function initializePosenet(){
    let poseOptions = {
        architecture: 'ResNet50', //'MobileNetV1' or 'ResNet50'
        imageScaleFactor: 0.3,
        outputStride: 16,
        flipHorizontal: false,
        minConfidence: 0.5,
        maxPoseDetections: 5,
        scoreThreshold: 0.5,
        nmsRadius: 20,
        detectionType: 'multiple', //'single' or 'multiple'
        inputResolution: 513,
        multiplier: 0.75,
        quantBytes: 2,
    };
    posenet = ml5.poseNet(video, poseOptions, poseModelLoaded);
    posenet.on('pose', gotPoses);
}

function initializeBrain(){
    let brainOptions = {
        inputs: 34, //17 pose skeleton points x and y = 17*2
        outputs: 24, //as many as number of poses, number of classifications
        task: 'classification',
        debug : true,
    }
    brain = ml5.neuralNetwork(brainOptions);
    
    if (fileState == "modelTraining") {
        brain.loadData('24PoseData.json', dataReady);
    } else if (fileState == "modelDeployed") {
        const modelInfo = {
            model: 'model2/model.json',
            metadata: 'model2/model_meta.json',
            weights: 'model2/model.weights.bin',
        };
        brain.load(modelInfo, brainLoaded); 
    }
}

function brainLoaded() {
    console.log('pose classification ready!');
    classifyPose();
}
  
function classifyPose() {
    if (pose) {
        if (pose.score > 0.75) {
            let inputs = [];
            for (let i = 0; i < pose.keypoints.length; i++) {
              let x = pose.keypoints[i].position.x;
              let y = pose.keypoints[i].position.y;
              inputs.push(x);
              inputs.push(y);
            }
            brain.classify(inputs, gotResult);
          } else {
              poseLabel = "unsure pose";
              poseLabelScore = 0.0;
              setTimeout(classifyPose, 200);
          }
    } else {
        poseLabel = "no pose";
        poseLabelScore = 0.0;
        setTimeout(classifyPose, 200);
    }
    
}
  
function gotResult(error, results) {
    console.log(results);

    if (results[0].confidence > 0.8) {
        poseLabel = results[0].label;
        poseLabelScore = results[0].confidence;
    } else if (results[0].confidence <= 0.8) {
        poseLabel = "unsure";
        poseLabelScore = results[0].confidence;
    }
    //console.log(results[0].confidence);
    setTimeout(classifyPose, 200);
}

function dataReady() {
    brain.normalizeData();
    brain.train({epochs: 200}, trainingFinished); 
}
  
function trainingFinished() {
    console.log('model trained');
    brain.save();
}

function poseModelLoaded() {
    console.log("poseNet ready");
}

function gotPoses(poses) {
    //console.log(poses);
    /*
        poses = [
            {
                pose: {score: 0.40012150148664755, keypoints: Array(17), nose: {…}, leftEye: {…}, rightEye: {…}, …},
                skeleton: []
            },
            {pose: {…}, skeleton: []},
        ];
    */
    if (poses.length > 0) {
        //you can also check confidence
        pose = poses[0].pose;
        skeleton = poses[0].skeleton;

        if (state == 'collecting' && pose.score > 0.75) {
            let inputs = [];
            for (let i = 0; i < pose.keypoints.length; i++) {
              let x = pose.keypoints[i].position.x;
              let y = pose.keypoints[i].position.y;
              inputs.push(x);
              inputs.push(y);
            }
            let target = [targetLabel];
            brain.addData(inputs, target);
        }
    }
}

function keyPressed() {
    if (fileState == "dataCollection") {
        if (key == 's') {
            brain.saveData();
         } else if (key == 'r') {
             targetLabel = poseInput.value();
             console.log(targetLabel);
     
             setTimeout(function() {
                 console.log('collecting');
                 state = "collecting";
                 setTimeout(function() {
                 console.log('not collecting');
                 state = "waiting";
                 }, 10000);
             }, 5000);
         }
    }
}


function draw() {
    background("white");
    
    translate(video.width*videoScaleVal, 0);
    scale(-1*videoScaleVal, videoScaleVal); //scale(-2, 2); 
    image(video, 0, 0);
    filter(INVERT);
    
    if (pose) {

        //console.log(pose.score);
        push();
        scale(-1, 1);
        textSize(48); 
        textStyle(BOLD);
        textAlign(CENTER, CENTER);
        //text(pose.score, -1*video.width*videoScaleVal/2, video.height*videoScaleVal/2, video.width, video.height);
        text(poseLabel, -1*video.width*videoScaleVal/2, video.height*videoScaleVal/2, video.width, video.height-10);
        text(poseLabelScore, -1*video.width*videoScaleVal/2, video.height*videoScaleVal/4, video.width, video.height-10);
        pop();

        //Draw all the 17 keypoints on the body
        for (let i = 0; i < pose.keypoints.length; i++) {
            let xPos = pose.keypoints[i].position.x;
            let yPos = pose.keypoints[i].position.y;
            fill("white");
            ellipse(xPos, yPos, 8, 8);
        }

        //Draw the skeleton that connects specific keypoints in the body
        for (let i = 0; i < skeleton.length; i++) {
            //skeleton is a 2D array for the two points that make a line of the skeleton
            let a = skeleton[i][0];
            let b = skeleton[i][1];
            push();
            stroke("white");
            strokeWeight(2);
            line(a.position.x, a.position.y, b.position.x, b.position.y);
            pop();
        }

        push();
        fill("#ca0020");
        ellipse(pose.nose.x, pose.nose.y, 16);
        rect(pose.rightWrist.x, pose.rightWrist.y, 16, 16);
        rect(pose.leftWrist.x, pose.leftWrist.y, 16, 16);
        pop();
    }
    
}