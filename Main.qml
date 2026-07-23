import QtQuick
import QtQuick.Layouts
import QtQuick.Controls.Basic
import QtQuick.Controls
import Qt.labs.qmlmodels

ApplicationWindow {
    id: window
    width: 640
    height: 480
    minimumWidth: 200
    minimumHeight: 250
    visible: true
    title: qsTr("Handwritten Digit Recognizer")
    visibility: Window.Maximized
    readonly property real scaleFactor: Math.min(window.width / 1024, window.height / 768)

    Rectangle {
        id: rectangle
        anchors.fill: parent
        color: "#1f1f1f"

        Rectangle {
            id: rectangle1
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.margins: 10
            height: 50
            color: "#ffffff"
            radius: 10
            border.width: 5

            Text {
                id: text4
                anchors.centerIn: parent
                text: qsTr("Handwritten Digit Recognizer - Made By A.Sh")
                font.pixelSize: 26
                font.bold: true
            }
        }

        GridLayout {
            id: grid
            anchors.top: rectangle1.bottom
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.margins: 10
            anchors.topMargin: 10
            anchors.bottomMargin: 20

            rows: 2
            columns: 2

            // --- ROW 1, COL 1 (Fixed Size) ---
            Rectangle {
                id: rectangle2
                Layout.preferredWidth: 420 * scaleFactor
                Layout.preferredHeight: 450 * scaleFactor
                color: "#5a5a5a"
                radius: 10

                Canvas {
                    id: drawingCanvas
                    anchors.top: parent.top
                    anchors.topMargin: 40
                    anchors.left: parent.left
                    anchors.leftMargin: 10
                    anchors.right: parent.right
                    anchors.rightMargin: 10
                    width: 400 * scaleFactor
                    height: 400 * scaleFactor


                    property var paths: []
                    property var currentPath: []

                    function clear() {
                        paths = []
                        currentPath = []
                        requestPaint()
                    }

                    onPaint: {
                        var ctx = getContext("2d")

                        ctx.fillStyle = "black"
                        ctx.fillRect(0, 0, width, height)

                        ctx.strokeStyle = "white"
                        ctx.lineWidth = 20 * scaleFactor
                        ctx.lineCap = "round"
                        ctx.lineJoin = "round"

                        for (var i = 0; i < paths.length; i++) {
                            var p = paths[i]
                            if (p.length < 2) continue
                            ctx.beginPath()
                            ctx.moveTo(p[0].x, p[0].y)
                            for (var j = 1; j < p.length; j++) {
                                ctx.lineTo(p[j].x, p[j].y)
                            }
                            ctx.stroke()
                        }

                        if (currentPath.length >= 2) {
                            ctx.beginPath()
                            ctx.moveTo(currentPath[0].x, currentPath[0].y)
                            for (var k = 1; k < currentPath.length; k++) {
                                ctx.lineTo(currentPath[k].x, currentPath[k].y)
                            }
                            ctx.stroke()
                        }
                    }

                    MouseArea {
                        id: mouseArea
                        anchors.fill: parent

                        onPressed: (mouse) => {
                            drawingCanvas.currentPath = [{x: mouse.x, y: mouse.y}]
                        }

                        onPositionChanged: (mouse) => {
                            if (pressed) {
                                drawingCanvas.currentPath.push({x: mouse.x, y: mouse.y})
                                drawingCanvas.requestPaint()
                            }
                        }

                        onReleased: (mouse) => {
                            drawingCanvas.paths.push(drawingCanvas.currentPath)
                            drawingCanvas.currentPath = []
                        }
                    }
                }

                Rectangle {
                    id: rectangle7
                    x: 10
                    y: 10
                    width: 400 * scaleFactor
                    height: 30
                    color: "#2d2d2d"
                    anchors.top: parent.top
                    anchors.topMargin: 10
                    anchors.left: parent.left
                    anchors.leftMargin: 10
                    anchors.right: parent.right
                    anchors.rightMargin: 10
                    Text {
                        id: text1
                        anchors.centerIn: parent
                        color: "#ffffff"
                        text: "Write a Number"
                        font.pixelSize: 20
                        font.bold: true
                    }
                }
            }

            // --- ROW 1, COL 2 (Fills Width, Height matches Row 1) ---
            Rectangle {
                id: result
                Layout.fillWidth: true
                Layout.preferredHeight: 450 * scaleFactor
                color: "#5a5a5a"
                radius: 10

                Rectangle {

                    anchors.top: parent.top
                    anchors.topMargin: 10
                    anchors.left: parent.left
                    anchors.leftMargin: 10
                    anchors.right: parent.right
                    anchors.rightMargin: 10

                    height: 30
                    color: "#2d2d2d"

                    Text {
                        anchors.centerIn: parent
                        text: "Results"
                        color: "white"
                        font.pixelSize:20
                        font.bold: true

                    }
                }
                Rectangle {
                    id: bestResult
                    x: 10* scaleFactor
                    y: 40
                    width: 300* scaleFactor
                    height: 300* scaleFactor
                    color: "#d9d9d9"
                    radius: 0
                    Text{
                        id: bestResultValue
                        anchors.verticalCenterOffset: -30   * scaleFactor
                        anchors.centerIn: parent
                        text:appManager.m_bestPredictedDigit>= 0
                             ? "Top Prediction: " + appManager.bestPredictedDigit
                             : "Draw a digit"
                        color:"black"
                        font.pixelSize:270 * scaleFactor
                        font.bold: true
                    }
                    Rectangle {
                        id: bestResultProbabilityBox
                        height:35
                        anchors.left: parent.left
                        anchors.leftMargin: 0
                        anchors.right: parent.right
                        anchors.rightMargin: 0
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: 0
                        color: "#080707"
                        radius: 0
                        Text{
                            id: bestResultProbability
                            anchors.centerIn: parent
                            text: "Confidence: " + (appManager.bestProb * 100).toFixed(3) + "%"
                            color:"white"
                            font.pixelSize:15 * scaleFactor
                            font.bold: true
                        }

                    }

                }

                Rectangle {
                    id: secondBestResult
                    x: 350* scaleFactor
                    y: 40
                    width: 250* scaleFactor
                    height: 250* scaleFactor
                    color: "#d9d9d9"
                    radius: 0
                    Text{
                        id: secondBestResultValue
                        anchors.verticalCenterOffset: -30* scaleFactor
                        anchors.centerIn: parent
                        text:appManager.m_secondBestPredictedDigit>= 0
                             ? "Second Best Prediction: " + appManager.secondBestPredictedDigit
                             : "Draw a digit"
                        color:"black"
                        font.pixelSize:200 * scaleFactor
                        font.bold: true
                    }
                    Rectangle {
                        id: secondBestResultProbabilityBox
                        height:35
                        anchors.left: parent.left
                        anchors.leftMargin: 0
                        anchors.right: parent.right
                        anchors.rightMargin: 0
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: 0
                        color: "#080707"
                        radius: 0
                        Text{
                            id: secondBestResultProbability
                            anchors.centerIn: parent
                            text: "Confidence: " + (appManager.secondBestProb * 100).toFixed(3) + "%"
                            color:"white"
                            font.pixelSize:15 * scaleFactor
                            font.bold: true
                        }

                    }

                }

                Rectangle {
                    id: thirdBestResult
                    x: 640 * scaleFactor
                    y: 40
                    width: 200* scaleFactor
                    height: 200* scaleFactor
                    color: "#d9d9d9"
                    radius: 0
                    Text{
                        id: thirdBestResultValue
                        anchors.verticalCenterOffset: -30* scaleFactor
                        anchors.centerIn: parent
                        text:appManager.m_thirdBestPredictedDigit>= 0
                             ? "Third Best Prediction: " + appManager.thirdBestPredictedDigit
                             : "Draw a digit"
                        color:"black"
                        font.pixelSize:150 * scaleFactor
                        font.bold: true
                    }
                    Rectangle {
                        id: thirdBestResultProbabilityBox
                        height:35
                        anchors.left: parent.left
                        anchors.leftMargin: 0
                        anchors.right: parent.right
                        anchors.rightMargin: 0
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: 0
                        color: "#080707"
                        radius: 0
                        Text{
                            id: thirdBestResultProbability
                            anchors.centerIn: parent
                            text: "Confidence: " + (appManager.thirdBestProb * 100).toFixed(3) + "%"
                            color:"white"
                            font.pixelSize:15 * scaleFactor
                            font.bold: true
                        }

                    }

                }
                TableView {
                    anchors.left: parent.left
                    anchors.leftMargin: 10
                    anchors.right: parent.right
                    anchors.rightMargin: 10
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: 3


                    width: 1000
                    height: 70

                    columnSpacing: 1
                    rowSpacing: 1
                    clip: true

                    model: TableModel {
                        TableModelColumn { display: "label"; }
                        TableModelColumn { display: "v0" }
                        TableModelColumn { display: "v1" }
                        TableModelColumn { display: "v2" }
                        TableModelColumn { display: "v3" }
                        TableModelColumn { display: "v4" }
                        TableModelColumn { display: "v5" }
                        TableModelColumn { display: "v6" }
                        TableModelColumn { display: "v7" }
                        TableModelColumn { display: "v8" }
                        TableModelColumn { display: "v9" }

                        rows: [
                            { "label": "NUMBERS", "v0": 0, "v1": 1, "v2": 2, "v3": 3, "v4": 4, "v5": 5, "v6": 6, "v7": 7, "v8": 8, "v9": 9 },
                            { "label": "PROBABILITY", "v0": "0%", "v1": "0%", "v2": "0%", "v3": "0%", "v4": "0%", "v5": "0%", "v6": "0%", "v7": "0%", "v8": "0%", "v9": "0%" }
                        ]
                    }

                    delegate: Rectangle {
                        implicitWidth: 80
                        implicitHeight: 30
                        color: "#333333"

                        Text {
                            anchors.centerIn: parent
                            text: display
                            color: "white"
                            font.pixelSize: 12
                        }
                    }
                }
                Rectangle{
                    anchors.top: parent.top
                    anchors.topMargin: 40
                    anchors.right: parent.right
                    anchors.rightMargin: 10
                    anchors.left: parent.left
                    anchors.leftMargin: 900* scaleFactor

                    color: "#333333"
                    height:100* scaleFactor


                    ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 10
                            spacing: 10
                            layoutDirection: Qt.LeftToRight
                            Text {
                                text: "Model Evaluation"
                                color: "white"
                                font.pixelSize: 15
                                font.bold: true
                                Layout.alignment: Qt.AlignHCenter
                            }

                            Text {
                                text:appManager.m_modelPerformance>= 0
                                     ? "Correct : " + appManager.thirdBestPredictedDigit + " %"
                                     : "Model has not been evaluated"
                                color: "white"
                                font.pixelSize: 15
                                font.bold: true
                                Layout.alignment: Qt.AlignHCenter
                            }

                            Text {
                                text: "70000/80000"
                                color: "white"
                                font.pixelSize: 15
                                font.bold: true
                                Layout.alignment: Qt.AlignHCenter
                            }
                        }
                }
            }

            // --- ROW 2, COL 1 (Fixed Width, Fills Remaining Height) ---
            Rectangle {
                id: rectangle4
                Layout.preferredWidth: 420 * scaleFactor
                Layout.fillHeight: true
                color: "#5a5a5a"
                radius: 10

                Button {
                    id: recognizeButton
                    text: "Recognize Number"
                    anchors.top: parent.top
                    anchors.topMargin: 10 * scaleFactor
                    anchors.left: parent.left
                    anchors.leftMargin: 5
                    anchors.right: parent.right
                    anchors.rightMargin: 5

                    height: 35* scaleFactor
                    font.bold: true
                    font.pixelSize: 18
                    palette.buttonText: "white"

                    background: Rectangle {
                        radius: 5
                        color: recognizeButton.pressed ? "#03a306" :
                            (recognizeButton.hovered ? "#027d04" : "#03a306")
                        Behavior on color { ColorAnimation { duration: 100 } }
                    }
                    onClicked:{
                        var pixels =canvas.getNormalizedPixels();
                        appManager.predictedFromPixels(pixels);
                    }
                }
                Button {
                    id: clearButton
                    text: "Clear Number"
                    anchors.top: parent.top
                    anchors.topMargin: 50 * scaleFactor
                    anchors.left: parent.left
                    anchors.leftMargin: 5
                    anchors.right: parent.right
                    anchors.rightMargin: 5
                    height: 35 * scaleFactor
                    font.bold: true
                    font.pixelSize: 18
                    palette.buttonText: "white"

                    background: Rectangle {
                        radius: 5
                        color: clearButton.pressed ? "#db0909" :
                            (clearButton.hovered ? "darkred" : "#db0909")
                        Behavior on color { ColorAnimation { duration: 100 } }
                    }
                    onClicked:{
                        drawingCanvas.clear()
                        appManager.clearPrediction();
                    }
                }
                Button {
                    id: trainModelButton
                    text: "Train Model"
                    anchors.top: parent.top
                    anchors.topMargin: 92 * scaleFactor
                    anchors.left: parent.left
                    anchors.leftMargin: 5

                    width: 200 * scaleFactor
                    height: 64 * scaleFactor
                    font.bold: true
                    font.pixelSize: 18
                    palette.buttonText: "white"

                    background: Rectangle {
                        radius: 5
                        color: trainModelButton.pressed ? "#3b3b3b" :
                            (trainModelButton.hovered ? "#000000" : "#242424")
                        Behavior on color { ColorAnimation { duration: 100 } }
                    }
                    onClicked:{
                        appManager.trainModel();
                    }
                }
                Button {
                    id: resetModelButton
                    text: "Reset Model"
                    anchors.top: parent.top
                    anchors.topMargin: 160 * scaleFactor
                    anchors.left: parent.left
                    anchors.leftMargin: 5

                    width: 200 * scaleFactor
                    height: 64 * scaleFactor
                    font.bold: true
                    font.pixelSize: 18
                    palette.buttonText: "white"
                    background: Rectangle {
                        radius: 5
                        color: resetModelButton.pressed ? "#3b3b3b" :
                            (resetModelButton.hovered ? "#000000" : "#242424")
                        Behavior on color { ColorAnimation { duration: 100 } }
                    }
                    onClicked:{
                        appManager.resetModel();
                    }
                }
                Button {
                    id: evaluateButton
                    text: "Evaluate Model"
                    anchors.top: parent.top
                    anchors.topMargin: 92 * scaleFactor
                    anchors.right: parent.right
                    anchors.rightMargin: 5
                    width: 200 * scaleFactor
                    height: 64* scaleFactor
                    font.bold: true
                    font.pixelSize: 18
                    palette.buttonText: "white"
                    background: Rectangle {
                        radius: 5
                        color: evaluateButton.pressed ? "#3b3b3b" :
                            (evaluateButton.hovered ? "#000000" : "#242424")
                        Behavior on color { ColorAnimation { duration: 100 } }
                    }
                    onClicked:{
                        appManager.evaluateModel();
                    }
                }
                Button {
                    id: myButton6
                    text: "Click Me"
                    anchors.top: parent.top
                    anchors.topMargin: 160 * scaleFactor
                    anchors.right: parent.right
                    anchors.rightMargin: 5
                    width: 200 * scaleFactor
                    height: 64* scaleFactor
                    font.bold: true
                    font.pixelSize: 18
                    palette.buttonText: "white"
                    background: Rectangle {
                        radius: 5
                        color: myButton6.pressed ? "#3b3b3b" :
                            (myButton6.hovered ? "#000000" : "#242424")
                        Behavior on color { ColorAnimation { duration: 100 } }
                    }
                }
            }

            // --- ROW 2, COL 2 (Fills Remaining Width AND Height) ---
            Rectangle {
                id: rectangle3
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: "#5a5a5a"
                radius: 10

                Text {
                    id: text2
                    x: 18
                    y: 16
                    width: 1280
                    height: 311
                    color: "#ffffff"
                    text: "Explain the buttons"
                    font.pixelSize: 12
                    minimumPointSize: 16
                    minimumPixelSize: 17
                }
            }
        }
    }
}