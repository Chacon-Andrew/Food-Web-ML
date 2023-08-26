import * as tf from '@tensorflow/tfjs';
import { Component, useState } from 'react';

function Model() {

    const [file, setFile] = useState();

    function handleChange(e) {
        console.log(e.target.files);
        setFile(URL.createObjectURL(e.target.files[0]));
    }

    async function loadModel() {
        this.model = await tf.loadGraphModel("https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_130_224/classification/2/default/1", { fromTFHub: true });
    }

    async function predict(ImageData) {
        const pred = await tf.tidy(() => {
            let img = tf.FromPixels(ImageData, 1);
            img = img.reshape([1, 28, 28, 1]);

            const output = this.model.predict(img);

            this.predictions = Array.from(output.dataSync());
        });
        console.log(this.predictions)
    }

    return(
        <div>
            <h1>Hello</h1>
            <input type='file' onChange={handleChange}/>
            <img src={file} style={{width: '500px', height: '500px'}}/>
        </div>  
    );
}

export default Model