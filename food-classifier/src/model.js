import * as tf from '@tensorflow/tfjs';
import { Component } from 'react';

class Model extends Component {

    async loadModel() {
        this.model = await tf.loadGraphModel("https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_130_224/classification/2/default/1", { fromTFHub: true });
    }

    async predict(ImageData) {
        const pred = await tf.tidy(() => {
            let img = tf.FromPixels(ImageData, 1);
            img = img.reshape([1, 28, 28, 1]);

            const output = this.model.predict(img);

            this.predictions = Array.from(output.dataSync());
        });
    }

    render() {
        <H1>Hello</H1>
    }
}

export default Model