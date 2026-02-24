let trainingInterval = null
let datasetUploaded = false

const overlay =
document.getElementById("trainingOverlay")

const overlayBar =
document.getElementById("overlayBar")

const overlayPercent =
document.getElementById("overlayPercent")

const trainBtn =
document.getElementById("trainBtn")


async function uploadDataset()
{

const fileInput =
document.getElementById("datasetZip")

if(fileInput.files.length===0)
{
alert("Select zip first")
return
}

const formData =
new FormData()

formData.append(
"file",
fileInput.files[0]
)

document.getElementById("uploadStatus").innerText =
"Uploading..."

try
{

const res =
await fetch(
"/upload_dataset",
{
method:"POST",
body:formData
}
)

const data =
await res.json()

if(data.status==="success")
{

datasetUploaded = true

document.getElementById("uploadStatus").innerText =
"Upload success"


trainBtn.disabled = false

trainBtn.classList.remove(
"bg-gray-400",
"cursor-not-allowed"
)

trainBtn.classList.add(
"bg-green-500",
"hover:bg-green-600"
)

}
else
{

document.getElementById("uploadStatus").innerText =
"Upload failed"

}

}
catch(err)
{

document.getElementById("uploadStatus").innerText =
"Upload error"

}

}

async function startTraining()
{

if(!datasetUploaded)
{
alert("Upload dataset first!")
return
}

overlay.classList.remove("hidden")

trainBtn.disabled=true

trainBtn.innerText="Training..."


const params =
{

model:
document.getElementById("model").value,

epochs:
document.getElementById("epochs").value,

lr:
document.getElementById("lr").value,

dropout:
document.getElementById("dropout").value,

fine_tune:
document.getElementById("fine_tune").checked

}


await fetch(
"/train",
{
method:"POST",
headers:
{
"Content-Type":
"application/json"
},
body:
JSON.stringify(params)
}
)


trainingInterval =
setInterval(
checkStatus,
1000
)

}

async function checkStatus()
{

const res =
await fetch("/training_status")

const data =
await res.json()


const percent =
data.progress || 0


overlayBar.style.width =
percent + "%"

overlayPercent.innerText =
percent + "%"



if(data.status==="finished")
{

clearInterval(trainingInterval)

overlay.classList.add("hidden")

trainBtn.disabled=false

trainBtn.innerText="Start Training"

showResult(data)

}

}

function showResult(data)
{

document.getElementById("metrics")
.classList.remove("hidden")

document.getElementById("plots")
.classList.remove("hidden")

document.getElementById("downloadSection")
.classList.remove("hidden")


document.getElementById("accuracy")
.innerText =
"Accuracy: "+
data.metrics.accuracy.toFixed(4)


document.getElementById("precision")
.innerText =
"Precision: "+
data.metrics.precision.toFixed(4)


document.getElementById("recall")
.innerText =
"Recall: "+
data.metrics.recall.toFixed(4)


document.getElementById("f1")
.innerText =
"F1 Score: "+
data.metrics.f1.toFixed(4)



document.getElementById("confusionImg")
.src =
"/"+data.plots.confusion+
"?t="+new Date().getTime()


document.getElementById("accuracyImg")
.src =
"/"+data.plots.accuracy+
"?t="+new Date().getTime()

}