# onnxruntime-react-native-jsi

Experimental React Native JSI implement for onnxruntime

## Installation

```sh
npm install onnxruntime-react-native-jsi

# Or alias
npm install onnxruntime-react-native@npm:onnxruntime-react-native-jsi
```

## Usage


```js
// Just use like normal `onnxruntime-react-native`
import { InferenceSession, Tensor } from 'onnxruntime-react-native-jsi';

const model = await InferenceSession.create('path/to/model.onnx')
```


## Contributing

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## License

MIT

---

Made with [create-react-native-library](https://github.com/callstack/react-native-builder-bob)
