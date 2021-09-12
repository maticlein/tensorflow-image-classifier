let net;

const imgEl = document.getElementById('img');
const descEl = document.getElementById('descripcion_imagen');
const webcamElement = document.getElementById('webcam');

async function app(){
    net = await mobilenet.load();
    var result = await net.classify(imgEl);
    console.log(result);
    displayImagePrediction();
    webcam = await tf.data.webcam(webcamElement);
    
    while(true){
        const img = await webcam.capture();
        const result = await net.classify(img);
        document.getElementById('console').innerHTML = 'prediction:' + result[0].className + ' ' + 'probability:' + result[0].probability
        img.dispose();
        await tf.nextFrame();
    }
}

imgEl.onload = async function(){
    displayImagePrediction();
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