async function runExample() {
  // Corrected array initialization with Float32Array and the proper length.
  var x = new Float32Array(36);

  // Assign values from form fields.
  for (let i = 0; i < 36; i++) {
    let element = document.getElementById(`box${i + 1}`);
    if (element) {
      x[i] = parseFloat(element.value);
    } else {
      console.warn(`Element box${i + 1} not found.`);
    }
  }

  // Create a tensor from the input data.
  let tensorX = new ort.Tensor('float32', x, [1, 36]);
  let feeds = { float_input: tensorX };

  // Load the ONNX model.
  let session = await ort.InferenceSession.create('Student_Dropout.onnx');
  
  // Run inference with the model.
  let result = await session.run(feeds);
  let outputData = result.variable.data;

  // Format the output to two decimal places.
  outputData = parseFloat(outputData).toFixed(2);

  // Display the inference results.
  let predictions = document.getElementById('predictions');
  predictions.innerHTML = `
    <hr>
    <h3>Inference Results</h3>
    <table border="1">
      <tr>
        <th>Rate of Student Dropout</th>
        <td>${outputData}</td>
      </tr>
    </table>
  `;
}
