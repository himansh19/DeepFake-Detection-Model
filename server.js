const h = require("./uploads/h");
const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const multer = require('multer');
const app = express();
const port = 3000;
const upload = multer({ dest: 'uploads/' });
const cors = require('cors');
app.use(cors());
app.use(bodyParser.json());

app.post('/predict',upload.single('videoFile'), (req, res) => {
    const videoFile = req.file;
    if (!videoFile) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    // Spawn a Python child process
    const pythonProcess = spawn('python', ['predict.py', videoFile.path]);
    // Capture output from the Python script
    pythonProcess.stdout.on('data', (data) => {
      const result = data.toString().trim();
      res.json({ prediction: result });
    });
  
    // Handle errors
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Error: ${data}`);
      res.status(500).send('Internal Server Error');
    });
  });
  

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
