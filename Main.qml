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

                Canvas {
                    id: drawingCanvas

                    // 1. DYNAMIC SIZING: Fill parent container or use relative anchors
                    anchors.fill: parent
                    anchors.top: parent.top
                    anchors.topMargin: 30
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: 10
                    anchors.left: parent.left
                    anchors.leftMargin: 10
                    anchors.right: parent.right
                    anchors.rightMargin: 10

                    property var paths: []
                    property var currentPath: []
                    property color strokeColor: "#FFFFFF"
                    property real currentLineWidth: 18 * scaleFactor
                    // Store previous size to calculate scaling factors when resized
                    property real lastWidth: width
                    property real lastHeight: height

                    // Trigger prediction manually via button click
                    function triggerPrediction() {
                        var captureX = mouseArea.x;
                        var captureY = mouseArea.y;
                        var captureWidth = mouseArea.width;
                        var captureHeight = mouseArea.height;

                        drawingCanvas.grabToImage(function(result) {
                            appManager.predictFromImage(result.image);
                        }, Qt.rect(captureX, captureY, captureWidth, captureHeight));
                    }

                    function clear() {
                        paths = []
                        currentPath = []
                        requestPaint()
                        appManager.clearPrediction()
                    }

                    // 2. RESIZE HANDLING: Scale stroke coordinates proportionally when screen resizes
                    onWidthChanged: rescalePaths()
                    onHeightChanged: rescalePaths()

                    function rescalePaths() {
                        if (lastWidth <= 0 || lastHeight <= 0) {
                            lastWidth = width;
                            lastHeight = height;
                            return;
                        }

                        var scaleX = width / lastWidth;
                        var scaleY = height / lastHeight;

                        // Recalculate stored paths relative to new dimensions
                        for (var i = 0; i < paths.length; i++) {
                            for (var j = 0; j < paths[i].length; j++) {
                                paths[i][j].x *= scaleX;
                                paths[i][j].y *= scaleY;
                            }
                        }

                        lastWidth = width;
                        lastHeight = height;
                        requestPaint();
                    }

                    onPaint: {
                        var ctx = getContext("2d")

                        // Draw background
                        ctx.fillStyle = "black"
                        ctx.fillRect(0, 0, width, height)

                        // Adjust line width proportionally relative to canvas size
                        ctx.strokeStyle = "white"
                        ctx.lineWidth = Math.min(width, height) * 0.10
                        ctx.lineCap = "round"
                        ctx.lineJoin = "round"

                        // Render completed paths
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

                        // Render current active path
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
                        anchors.margins: 50 * scaleFactor

                        Rectangle {
                            anchors.fill: parent
                            // This is only a writing guide.  Hide it before a
                            // recognizer-button click so it cannot enter the
                            // Canvas snapshot as a grey rectangular "digit".
                            visible: mouseArea.containsMouse || mouseArea.pressed
                            color: mouseArea.containsMouse ? "#10FFFFFF" : "transparent"
                            border.color: mouseArea.containsMouse ? "#00FF00" : "#555555"
                            border.width: 2
                            radius: 4
                        }

                        Rectangle {
                                id: innerActiveRect
                                anchors.fill: parent
                                anchors.margins: 60 * scaleFactor // Makes it smaller than outer box

                                // Fully invisible when not writing
                                visible: mouseArea.pressed

                                color: "#20FF0000" // Light semi-transparent red fill
                                border.color: "#FF3333" // Bright red border
                                border.width: 2
                                radius: 4
                        }

                        onPressed: (mouse) => {
                            // Map local MouseArea coords (mouse.x, mouse.y) to Canvas coords
                            var pt = mapToItem(drawingCanvas, mouse.x, mouse.y)
                            drawingCanvas.currentPath = [{x: pt.x, y: pt.y}]
                        }

                        onPositionChanged: (mouse) => {
                            if (pressed) {
                                var pt = mapToItem(drawingCanvas, mouse.x, mouse.y)
                                drawingCanvas.currentPath.push({x: pt.x, y: pt.y})
                                drawingCanvas.requestPaint()
                            }
                        }

                        onReleased: (mouse) => {
                            drawingCanvas.paths.push(drawingCanvas.currentPath)
                            drawingCanvas.currentPath = []
                        }
                    }
                }

                // This is the exact 28x28 image sent to the CNN, enlarged
                // without smoothing.  It lets the user verify that their
                // drawing survived cropping, centering, and resizing.
                Rectangle {
                    id: modelInputPreview
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    anchors.rightMargin: 14 * scaleFactor
                    anchors.bottomMargin: 14 * scaleFactor
                    width: 106 * scaleFactor
                    height: 126 * scaleFactor
                    z: 2
                    radius: 5
                    color: "#dd101010"
                    border.color: "#cccccc"
                    border.width: 1
                    visible: appManager.processedInputPreview !== ""

                    Text {
                        anchors.top: parent.top
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.topMargin: 5 * scaleFactor
                        text: "CNN input"
                        color: "white"
                        font.pixelSize: 12 * scaleFactor
                        font.bold: true
                    }

                    Image {
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: 7 * scaleFactor
                        width: 92 * scaleFactor
                        height: 92 * scaleFactor
                        source: appManager.processedInputPreview
                        fillMode: Image.PreserveAspectFit
                        smooth: false
                        mipmap: false
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
                    anchors.left: parent.left
                    anchors.leftMargin: 10 * scaleFactor
                    anchors.top: parent.top
                    anchors.topMargin: 50 * scaleFactor
                    width: (parent.width - 40 * scaleFactor) * 0.40
                    height: Math.min(parent.height * 0.60, width)
                    color: "#d9d9d9"
                    radius: 0
                    Text{
                        id: bestResultValue
                        anchors.verticalCenterOffset: -30   * scaleFactor
                        anchors.centerIn: parent
                        text:appManager.bestPredictedDigit>= 0
                             ? appManager.bestPredictedDigit
                             : " "
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
                    anchors.left: bestResult.right
                    anchors.leftMargin: 10 * scaleFactor
                    anchors.top: bestResult.top
                    width: (parent.width - 40 * scaleFactor) * 0.33
                    height: bestResult.height * 0.84
                    color: "#d9d9d9"
                    radius: 0
                    Text{
                        id: secondBestResultValue
                        anchors.verticalCenterOffset: -30* scaleFactor
                        anchors.centerIn: parent
                        text:appManager.secondBestPredictedDigit>= 0
                             ? appManager.secondBestPredictedDigit
                             : " "
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
                    anchors.left: secondBestResult.right
                    anchors.leftMargin: 10 * scaleFactor
                    anchors.right: parent.right
                    anchors.rightMargin: 10 * scaleFactor
                    anchors.top: bestResult.top
                    height: bestResult.height * 0.68
                    color: "#d9d9d9"
                    radius: 0
                    Text{
                        id: thirdBestResultValue
                        anchors.verticalCenterOffset: -30* scaleFactor
                        anchors.centerIn: parent
                        text:appManager.thirdBestPredictedDigit>= 0
                             ? appManager.thirdBestPredictedDigit
                             : " "
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
                    id: confidenceTable
                    anchors.left: parent.left
                    anchors.leftMargin: 10
                    anchors.right: parent.right
                    anchors.rightMargin: 10
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: 3


                    height: Math.max(58, 72 * scaleFactor)

                    columnSpacing: 1
                    rowSpacing: 1
                    clip: true
                    columnWidthProvider: function() { return (confidenceTable.width - 10) / 11 }
                    rowHeightProvider: function() { return (confidenceTable.height - 2) / 2 }

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
                        implicitWidth: confidenceTable.columnWidthProvider(column)
                        implicitHeight: confidenceTable.rowHeightProvider(row)
                        color: row === 0 ? "#3b3b3b" : "#292929"
                        border.color: "#4b4b4b"
                        border.width: 1

                        Text {
                            anchors.centerIn: parent
                            text: {
                                if (column === 0)
                                    return row === 0 ? "DIGIT" : "CONF."
                                if (row === 0)
                                    return column - 1
                                const values = appManager.probabilities
                                const probability = values.length === 10 ? values[column - 1] : 0
                                return (probability * 100).toFixed(1) + "%"
                            }
                            color: "white"
                            font.pixelSize: Math.max(9, Math.min(13, parent.height * 0.42))
                            font.bold: row === 0
                        }
                    }
                }
                Rectangle{
                    anchors.top: parent.top
                    anchors.topMargin: 6 * scaleFactor
                    anchors.right: parent.right
                    anchors.rightMargin: 10 * scaleFactor
                    width: Math.min(190 * scaleFactor, parent.width * 0.28)
                    color: "#333333"
                    height:36 * scaleFactor


                    ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 4 * scaleFactor
                            spacing: 0
                            Text {
                                text: "Model Evaluation"
                                color: "white"
                                font.pixelSize: Math.max(9, 11 * scaleFactor)
                                font.bold: true
                                Layout.fillWidth: true
                                horizontalAlignment: Text.AlignHCenter
                            }

                            Text {

                                property bool isEvaluated: (appManager?.modelPerformance?? 0)>0
                                text:isEvaluated
                                     ? appManager.modelPerformance.toFixed(2) + "%"
                                     : "Not evaluated"

                                color: "white"
                                font.pixelSize: Math.max(9, 11 * scaleFactor)
                                font.bold: true
                                Layout.fillWidth: true
                                horizontalAlignment: Text.AlignHCenter

                                wrapMode: isEvaluated? Text.NoWrap : Text.WordWrap
                                fontSizeMode: Text.Fit
                                minimumPixelSize:8
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
                        drawingCanvas.triggerPrediction()
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
                        console.log(">>> [QML] Button clicked, calling trainModel()...")
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
                        console.log(">>> [QML] Button clicked, calling evaluateModel()...")
                        appManager.evaluateModel();
                    }
                }
                Rectangle {
                    radius: 5
                    color: "#3b3b3b"
                    width: 200 * scaleFactor
                    height: 64* scaleFactor

                    anchors.top: parent.top
                    anchors.topMargin: 160 * scaleFactor
                    anchors.right: parent.right
                    anchors.rightMargin: 5
                    Text {
                        text: appManager?.actionDone ?? ""
                        width: parent.width - 16 * scaleFactor
                        anchors.centerIn: parent
                        font.bold: true
                        font.pixelSize: 18 * scaleFactor
                        color: "#12ff22"
                        wrapMode: Text.WordWrap
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
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

                readonly property bool compactCards: width < 620 * scaleFactor

                Rectangle {
                    id: teachCard
                    anchors.left: parent.left
                    anchors.leftMargin: 12 * scaleFactor
                    anchors.top: parent.top
                    anchors.topMargin: 12 * scaleFactor
                    width: parent.compactCards ? parent.width - 24 * scaleFactor : parent.width * 0.43
                    height: parent.compactCards ? (parent.height - 36 * scaleFactor) * 0.48
                                                : parent.height - 24 * scaleFactor
                    radius: 8
                    color: "#393939"
                    border.color: "#747474"
                    border.width: 1

                    Column {
                    anchors.fill: parent
                    anchors.margins: 12 * scaleFactor
                    spacing: 7 * scaleFactor

                    Text {
                        text: "Teach your handwriting"
                        color: "white"
                        font.bold: true
                        font.pixelSize: Math.max(13, 17 * scaleFactor)
                    }

                    Text {
                        width: parent.width
                        text: "Draw a digit, press Recognize Number, choose its correct value, then press Teach. Add 3–5 examples for digits the model confuses, such as 6 or 8."
                        color: "#dddddd"
                        wrapMode: Text.WordWrap
                        font.pixelSize: Math.max(10, 12 * scaleFactor)
                    }

                    Row {
                        spacing: 10 * scaleFactor

                        SpinBox {
                            id: correctDigitSelector
                            from: 0
                            to: 9
                            value: 6
                            editable: true
                            width: 72 * scaleFactor
                            height: 34 * scaleFactor
                        }

                        Button {
                            text: "Teach digit"
                            width: Math.max(100, 122 * scaleFactor)
                            height: 34 * scaleFactor
                            font.bold: true
                            enabled: appManager.bestPredictedDigit >= 0
                            onClicked: appManager.learnLastDigit(correctDigitSelector.value)
                        }
                    }
                    }
                }

                Rectangle {
                    id: programInfoCard
                    anchors.top: parent.compactCards ? teachCard.bottom : parent.top
                    anchors.topMargin: 12 * scaleFactor
                    anchors.left: parent.compactCards ? parent.left : teachCard.right
                    anchors.leftMargin: 12 * scaleFactor
                    anchors.right: parent.right
                    anchors.rightMargin: 12 * scaleFactor
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: 12 * scaleFactor
                    radius: 8
                    color: "#303030"
                    border.color: "#626262"
                    border.width: 1

                    Column {
                        anchors.fill: parent
                        anchors.margins: 12 * scaleFactor
                        spacing: 7 * scaleFactor

                        Text {
                            text: "About this program"
                            color: "white"
                            font.bold: true
                            font.pixelSize: Math.max(13, 17 * scaleFactor)
                        }

                        Text {
                            width: parent.width
                            text: "This Qt/C++ application recognises handwritten digits with a convolutional neural network. It trains on MNIST, evaluates on 10,000 test images, and can learn your own writing style through the teaching card."
                            color: "#dddddd"
                            wrapMode: Text.WordWrap
                            font.pixelSize: Math.max(10, 12 * scaleFactor)
                        }
                    }
                }
            }
        }
    }
    Rectangle {
        anchors.fill: parent
        color: "#80808080"
        anchors.centerIn: parent
        visible: appManager.isTraining || appManager.isEvaluating
        TapHandler {
                gesturePolicy: TapHandler.WithinBounds
            }
        // Hide the spinner once C++ emits modelEvaluationChanged
        Connections {
            target: appManager
            function isTrainingChanged() {
                busyIndicator.running = false
            }
            function modelEvaluationChanged() {
                busyIndicator.running = false
            }
        }
        TapHandler {
            gesturePolicy: TapHandler.WithinBounds
        }
        HoverHandler {
            blocking: true
        }
        ColumnLayout {
            anchors.centerIn: parent
            spacing: 20
            width: Math.min(parent.width * 0.6, 400 * scaleFactor)

            // 1. Text Status
            Text {
                text: appManager?.isTraining
                      ? "Training Progress: " + Math.round((appManager?.trainingProgress ?? 0) * 100) + "%"
                      : "Model is being evaluated..."
                color: "white"
                font.pixelSize: 22 * scaleFactor
                font.bold: true
                Layout.alignment: Qt.AlignHCenter
            }

            // 2. Busy Indicator Spinner
            BusyIndicator {
                id: busyIndicator
                running: overlay.visible
                Layout.alignment: Qt.AlignHCenter
            }

            // 3. Progress Bar (Only visible while training)
            ProgressBar {
                id: trainingProgressBar
                Layout.fillWidth: true
                Layout.preferredHeight: 16 * scaleFactor
                visible: appManager.isTraining

                from: 0.0
                to: 1.0
                value: appManager?.trainingProgress ?? 0.0

                // Dark track background
                background: Rectangle {
                    implicitWidth: 200
                    implicitHeight: 16 * scaleFactor
                    color: "#2a2a2a"
                    radius: 8
                    border.color: "#444444"
                    border.width: 1
                }

                // Green filled progress bar
                contentItem: Item {
                    implicitWidth: 200
                    implicitHeight: 16 * scaleFactor

                    Rectangle {
                        width: trainingProgressBar.visualPosition * parent.width
                        height: parent.height
                        radius: 8
                        color: "#12ff22" // Bright green progress fill

                        Behavior on width {
                            NumberAnimation { duration: 150 }
                        }
                    }
                }
            }
        }
    }
}
