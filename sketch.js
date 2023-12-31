let video; //stores webcam video
let posenet; //initialize with ml5 object poseNet
let brain; //initialize with ml5 object neuralNetwork
let fileState = "modelDeployed"; //"modelDeployed" "dataCollection" "modelTraining"
let table; //to store poseToAudio as csv
let rows; //to store reference to poseToAudio csv rows
let font;

let pose; //regularly stores the first pose in poses
let skeleton; //regularly stores the first skeleton in poses

let state = "waiting";
let targetLabel;
let poseInput; // to name poses being captured dynamically
let poseLabel = "unknown";
let currClip = "no clip playing";
let poseLabelScore = 0.0;

let videoScaleVal = 1;
let isBodyVisible = true;
let audioPlaying = false;
let clips = [24];
let clipPlayed = [24];

let poseNames = {
 "p_1": "arms down",
 "p_2": "",
 "p_3": "",
 "p_4": "",
 "p_5": "",
 "p_6": "",
 "p_7": "",
 "p_8": "",
 "p_9": "",
 "p_10": "",
 "p_11": "",
 "p_12": "",
 "p_13": "",
 "p_14": "",
 "p_15": "",
 "p_16": "",
 "p_17": "",
 "p_18": "",
 "p_19": "",
 "p_20": "",
 "p_21": "",
 "p_22": "",
 "p_23": "",
 "p_24": ""
};

function preload() {
    table = loadTable("poseToAudio.csv", "csv", "header");
    font = loadFont("Lugrasimo-Regular.ttf");
}

function setup() {
    createCanvas(windowWidth, windowWidth);
    noStroke();
    rectMode(CENTER);

    rows = table.getRows();
    video  = createCapture(VIDEO);
    video.hide();

    for (let r = 0; r < rows.length; r++) {
        let clipInRow = rows[r].get('clipName');
        let indexInRow = parseInt(rows[r].get('audioId'));
        let clipPath = "audioClips/" + clipInRow;
        clips[indexInRow] = createAudio(clipPath, () => {
            clips[indexInRow].autoplay(false);
            clips[indexInRow].noLoop();
            clips[indexInRow].hideControls();
          });
    }

    console.log(clips);

    if (fileState == "dataCollection") {
        poseInput = createInput('');
        poseInput.position(1*video.width, video.height*3);
        poseInput.size(400, 32);
    }

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
    if (!audioPlaying) {
        if (pose && isBodyVisible) {
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
                poseLabel = "unclear";
                poseLabelScore = 0.0;
                setTimeout(classifyPose, 200);
            }
        } else {
            poseLabel = "noPose";
            poseLabelScore = 0.0;
            setTimeout(classifyPose, 200);
        }
    } else {
        poseLabel = "paused";
        poseLabelScore = 0.0;
        setTimeout(classifyPose, 200);
    }
}
  
function gotResult(error, results) {

    if (error) {
        console.log(error);
        return;
    }

    //console.log(results);
    //console.log(results[0].confidence);
    if (results[0].confidence > 0.8) {
        poseLabel = results[0].label;
        poseLabelScore = results[0].confidence;
    } else if (results[0].confidence <= 0.8) {
        poseLabel = "unsure";
        poseLabelScore = results[0].confidence;
    }
    
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
    /* poses = [
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
    
    //UNSURE
    else {
        pose = null;
        skeleton = null;
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
    drawVideo();
    
    if (pose) {

        isBodyVisible = true;
        drawSkeleton();

        //Draw all the 17 keypoints on the body
        for (let i = 0; i < pose.keypoints.length; i++) {
            let xPos = pose.keypoints[i].position.x;
            let yPos = pose.keypoints[i].position.y;
            fill("#2c7bb6");
            ellipse(xPos, yPos, 8, 8);
            //console.log(xPos + " " + yPos);
            if (xPos < 100 || xPos > 1000) isBodyVisible = false;
            if (yPos < 50 || yPos > 670) isBodyVisible = false;
        }

        //console.log("In Frame: " + isBodyVisible);

        push();
        fill("#ca0020");
            ellipse(pose.nose.x, pose.nose.y, 16, 16);
            ellipse(pose.rightWrist.x, pose.rightWrist.y, 16, 16);
            ellipse(pose.leftWrist.x, pose.leftWrist.y, 16, 16);
        pop();

        //console.log(pose.score);
        drawGUIText()
    }

    if (poseLabel!="unsure" || poseLabel!="unclear" || poseLabel!="noPose" || poseLabel!="paused") {
        for (let r = 0; r < rows.length; r++) {
            let poseInRow = rows[r].get('poseLabel');
            if (poseInRow == poseLabel && !audioPlaying) {
                //console.log (rows[r].get('clipName'));
                let index = parseInt(rows[r].get('audioId'));
                let clipDuration = clips[index].duration()*1000;
                clips[index].play();
                audioPlaying = true;
                //console.log(clipDuration);
                currClip = rows[r].get('clipName');
                setTimeout(function() { 
                    audioPlaying = false; 
                    currClip = "no clip playing";
                }, 
                clipDuration + 1000);
                break;
            }
        }
    }
    
}

function drawVideo() {
    translate(video.width*videoScaleVal, 0);
    scale(-1*videoScaleVal, videoScaleVal); //scale(-2, 2); 
    image(video, 0, 0);
    filter(INVERT);

    push();
    rectMode(CORNERS);
    noFill(); stroke("#fdae61"); strokeWeight(4);
    rect(0, 0, video.width*videoScaleVal, video.height*videoScaleVal);
    pop();

    push();
    rectMode(CORNERS);
    fill('rgba(6, 40, 210, 0.1)'); noStroke();
    rect(0, 0, video.width*videoScaleVal, video.height*videoScaleVal);
    pop();
}

function drawSkeleton() {
    //Draw the skeleton that connects specific keypoints in the body
    for (let i = 0; i < skeleton.length; i++) {
        //skeleton is a 2D array for the two points that make a line of the skeleton
        let a = skeleton[i][0];
        let b = skeleton[i][1];
        push();
        stroke("#d7191c");
        strokeWeight(2);
        line(a.position.x, a.position.y, b.position.x, b.position.y);
        pop();
    }
}

function drawGUIText() {
    push();
        scale(-1, 1);
        fill("#d7191c");
        textSize(40); 
        textStyle(BOLD);
        textFont(font);
        textAlign(CENTER, CENTER);
        //text(pose.score, -1*video.width*videoScaleVal/2, video.height*videoScaleVal/2, video.width, video.height);
        if(poseLabel!="unsure" || poseLabel!="unclear" || poseLabel!="noPose" || poseLabel!="paused") {
            let label = poseNames.poseLabel;
        }
        text(poseLabel, -1*video.width*videoScaleVal/2, video.height*videoScaleVal/8, video.width, video.height-10);
        text(currClip, -1*video.width*videoScaleVal/2, video.height*videoScaleVal/1.1, video.width, video.height-10);
        //text(poseLabelScore, -1*video.width*videoScaleVal/2, video.height*videoScaleVal/3.25, video.width, video.height-10);
        pop();
}