let net;

const imgEl = document.getElementById('img');
const descEl = document.getElementById('descripcion_imagen');
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();

async function app(){
    net = await mobilenet.load();
    var result = await net.classify(imgEl);
    console.log(result);
    displayImagePrediction();
    webcam = await tf.data.webcam(webcamElement);
    
    while(true){
        const img = await webcam.capture();
        const result = await net.classify(img);
        const activation = net.infer(img, "conv_preds");
        var result2;
        try{
            result2 = await classifier.predictClass(activation);
        } catch(error){
            result2 = {};
        }

        const classes = ["Untrained", "Gatos", "Yo", "OK", "Rock"]
        
        document.getElementById('console').innerText = `
        prediction: ${result[0].className}\n
        probability: ${result[0].probability}
        `;

        try{
            document.getElementById('console2').innerText = `
            prediction: ${classes[result2.label]}\n
            probability: ${result2.confidences[result2.label]}
            `;
        ;
        } catch(error){
            document.getElementById("console2").innerText="Untrained";
        }
        
        img.dispose();
        await tf.nextFrame();
    }
}

imgEl.onload = async function(){
    displayImagePrediction();
}

async function addExample(classId){
    console.log('added example')
    const img = await webcam.capture();
    const activation = net.infer(img, true);
    classifier.addExample(activation, classId);
    img.dispose();
}

async function displayImagePrediction(){
    try{
        result = await net.classify(imgEl);
        descEl.innerHTML = JSON.stringify(result);
    } catch(error){

    }
}

count = 0;

async function cambiarImagen(){
    count = count + 1;
    imgEl.src = "https://picsum.photos/200/300?random=" + count;
    
}

app();