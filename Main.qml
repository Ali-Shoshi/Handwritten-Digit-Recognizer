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

                    anchors.fill: parent
                    anchors.top: parent.top
                    anchors.topMargin: 40
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
                    property real lastWidth: width
                    property real lastHeight: height

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

                        ctx.fillStyle = "black"
                        ctx.fillRect(0, 0, width, height)

                        ctx.strokeStyle = "white"
                        ctx.lineWidth = Math.min(width, height) * 0.10
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
                        anchors.margins: 50 * scaleFactor

                        Rectangle {
                            anchors.fill: parent
                            visible: mouseArea.containsMouse || mouseArea.pressed
                            color: mouseArea.containsMouse ? "#10FFFFFF" : "transparent"
                            border.color: mouseArea.containsMouse ? "#00FF00" : "#555555"
                            border.width: 2
                            radius: 4
                        }

                        Rectangle {
                                id: innerActiveRect
                                anchors.fill: parent
                                anchors.margins: 60 * scaleFactor

                                visible: mouseArea.pressed

                                color: "#20FF0000"
                                border.color: "#FF3333"
                                border.width: 2
                                radius: 4
                        }

                        onPressed: (mouse) => {
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
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.bottom: bestResultProbabilityBox.top
                        anchors.margins: 4 * scaleFactor
                        text:appManager.bestPredictedDigit>= 0
                             ? appManager.bestPredictedDigit
                             : " "
                        color:"black"
                        font.pixelSize: Math.max(18, parent.height * 0.9)
                        minimumPixelSize: 14
                        fontSizeMode: Text.Fit
                        font.bold: true
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
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
                            anchors.fill: parent
                            anchors.margins: 3 * scaleFactor
                            text: "Confidence: " + (appManager.bestProb * 100).toFixed(3) + "%"
                            color:"white"
                            font.pixelSize: Math.max(8, parent.height * 0.42)
                            minimumPixelSize: 8
                            fontSizeMode: Text.Fit
                            font.bold: true
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
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
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.bottom: secondBestResultProbabilityBox.top
                        anchors.margins: 4 * scaleFactor
                        text:appManager.secondBestPredictedDigit>= 0
                             ? appManager.secondBestPredictedDigit
                             : " "
                        color:"black"
                        font.pixelSize: Math.max(18, parent.height * 0.75)
                        minimumPixelSize: 14
                        fontSizeMode: Text.Fit
                        font.bold: true
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
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
                            anchors.fill: parent
                            anchors.margins: 3 * scaleFactor
                            text: "Confidence: " + (appManager.secondBestProb * 100).toFixed(3) + "%"
                            color:"white"
                            font.pixelSize: Math.max(8, parent.height * 0.42)
                            minimumPixelSize: 8
                            fontSizeMode: Text.Fit
                            font.bold: true
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
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
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.bottom: thirdBestResultProbabilityBox.top
                        anchors.margins: 4 * scaleFactor
                        text:appManager.thirdBestPredictedDigit>= 0
                             ? appManager.thirdBestPredictedDigit
                             : " "
                        color:"black"
                        font.pixelSize: Math.max(18, parent.height * 0.72)
                        minimumPixelSize: 14
                        fontSizeMode: Text.Fit
                        font.bold: true
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
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
                            anchors.fill: parent
                            anchors.margins: 3 * scaleFactor
                            text: "Confidence: " + (appManager.thirdBestProb * 100).toFixed(3) + "%"
                            color:"white"
                            font.pixelSize: Math.max(8, parent.height * 0.42)
                            minimumPixelSize: 8
                            fontSizeMode: Text.Fit
                            font.bold: true
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
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
            }

            // --- ROW 2, COL 1 (Fixed Width, Fills Remaining Height) ---
            Rectangle {
                id: rectangle4
                Layout.preferredWidth: 420 * scaleFactor
                Layout.fillHeight: true
                color: "#5a5a5a"
                radius: 10

                Rectangle {
                    id: responsiveControls
                    anchors.fill: parent
                    color: parent.color
                    radius: parent.radius
                    z: 10

                    MouseArea {
                        anchors.fill: parent
                        z: 0
                    }

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: Math.max(5, 9 * scaleFactor)
                        spacing: Math.max(4, 7 * scaleFactor)
                        z: 1

                        Button {
                            id: responsiveRecognizeButton
                            text: "Recognize Number"
                            Layout.fillWidth: true
                            Layout.preferredHeight: Math.max(28, responsiveControls.height * 0.15)
                            font.bold: true
                            font.pixelSize: Math.max(10, Math.min(18, height * 0.46))
                            palette.buttonText: "white"
                            background: Rectangle {
                                radius: 5
                                color: responsiveRecognizeButton.pressed ? "#03a306"
                                    : (responsiveRecognizeButton.hovered ? "#027d04" : "#03a306")
                            }
                            onClicked: drawingCanvas.triggerPrediction()
                        }

                        Button {
                            id: responsiveClearButton
                            text: "Clear Number"
                            Layout.fillWidth: true
                            Layout.preferredHeight: Math.max(28, responsiveControls.height * 0.15)
                            font.bold: true
                            font.pixelSize: Math.max(10, Math.min(18, height * 0.46))
                            palette.buttonText: "white"
                            background: Rectangle {
                                radius: 5
                                color: responsiveClearButton.pressed ? "#db0909"
                                    : (responsiveClearButton.hovered ? "darkred" : "#db0909")
                            }
                            onClicked: drawingCanvas.clear()
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            Layout.preferredHeight: Math.max(34, responsiveControls.height * 0.26)
                            spacing: Math.max(4, 7 * scaleFactor)

                            Button {
                                id: responsiveTrainButton
                                text: "Train Model"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                font.bold: true
                                font.pixelSize: Math.max(9, Math.min(17, height * 0.33))
                                palette.buttonText: "white"
                                background: Rectangle {
                                    radius: 5
                                    color: responsiveTrainButton.pressed ? "#3b3b3b"
                                        : (responsiveTrainButton.hovered ? "#000000" : "#242424")
                                }
                                onClicked: appManager.trainModel()
                            }

                            Button {
                                id: responsiveEvaluateButton
                                text: "Evaluate Model"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                font.bold: true
                                font.pixelSize: Math.max(9, Math.min(17, height * 0.33))
                                palette.buttonText: "white"
                                background: Rectangle {
                                    radius: 5
                                    color: responsiveEvaluateButton.pressed ? "#3b3b3b"
                                        : (responsiveEvaluateButton.hovered ? "#000000" : "#242424")
                                }
                                onClicked: appManager.evaluateModel()
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            Layout.preferredHeight: Math.max(34, responsiveControls.height * 0.26)
                            spacing: Math.max(4, 7 * scaleFactor)

                            Button {
                                id: responsiveResetButton
                                text: "Reset Model"
                                Layout.fillWidth: false
                                Layout.preferredWidth: responsiveControls.width * 0.28
                                Layout.fillHeight: true
                                font.bold: true
                                font.pixelSize: Math.max(9, Math.min(17, height * 0.33))
                                palette.buttonText: "white"
                                background: Rectangle {
                                    radius: 5
                                    color: responsiveResetButton.pressed ? "#3b3b3b"
                                        : (responsiveResetButton.hovered ? "#000000" : "#242424")
                                }
                                onClicked: appManager.resetModel()
                            }

                            Rectangle {
                                id: actionStatusCard
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                radius: 5
                                color: "#3b3b3b"

                                Text {
                                    id: actionStatusTitle
                                    anchors.top: parent.top
                                    anchors.left: parent.left
                                    anchors.right: parent.right
                                    anchors.topMargin: 3 * scaleFactor
                                    text: "Latest action"
                                    color: "#ffffff"
                                    font.bold: true
                                    font.pixelSize: Math.max(8, Math.min(13, parent.height * 0.25))
                                    horizontalAlignment: Text.AlignHCenter
                                }

                                Text {
                                    anchors.top: actionStatusTitle.bottom
                                    anchors.bottom: parent.bottom
                                    anchors.left: parent.left
                                    anchors.right: parent.right
                                    anchors.margins: 5 * scaleFactor
                                    text: appManager.actionDone === "" ? "Ready" : appManager.actionDone
                                    color: "#12ff22"
                                    font.bold: true
                                    font.pixelSize: Math.max(8, Math.min(15, parent.height * 0.28))
                                    minimumPixelSize: 8
                                    fontSizeMode: Text.Fit
                                    wrapMode: Text.WordWrap
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                            }
                        }
                    }
                }

                Button {
                    id: recognizeButton
                    visible: false
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
                    visible: false
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
                    visible: false
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
                    visible: false
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
                    visible: false
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
                        height: parent.height - 16 * scaleFactor
                        anchors.centerIn: parent
                        fontSizeMode: Text.Fit
                        minimumPixelSize: 10
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

                Rectangle {
                    id: teachCard
                    anchors.left: parent.left
                    anchors.leftMargin: 12 * scaleFactor
                    anchors.top: parent.top
                    anchors.topMargin: 12 * scaleFactor
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: 12 * scaleFactor
                    width: (parent.width - 36 * scaleFactor) * 0.23
                    radius: 8
                    color: "#393939"
                    border.color: "#747474"
                    border.width: 1

                    Column {
                    anchors.fill: parent
                    anchors.margins: 12 * scaleFactor
                    spacing: 7 * scaleFactor

                    Text {
                        text: "Teach model"
                        color: "white"
                        font.bold: true
                        font.pixelSize: Math.max(13, 14 * scaleFactor)
                    }

                    Text {
                        visible: false
                        width: parent.width
                        text: "Draw a digit, press Recognize Number, choose its correct value, then press Teach. Add 3–5 examples for digits the model confuses, such as 6 or 8."
                        color: "#dddddd"
                        wrapMode: Text.WordWrap
                        font.pixelSize: Math.max(10, 12 * scaleFactor)
                    }

                    Text {
                        width: parent.width
                        text: "Recognize, select the correct digit, then teach it. Add 3 to 5 samples."
                        color: "#dddddd"
                        wrapMode: Text.WordWrap
                        font.pixelSize: Math.max(10, 12 * scaleFactor)
                    }

                    Column {
                        width: parent.width
                        spacing: 5 * scaleFactor

                        SpinBox {
                            id: correctDigitSelector
                            from: 0
                            to: 9
                            value: 6
                            editable: true
                            width: parent.width
                            height: 30 * scaleFactor
                        }

                        Button {
                            text: "Teach digit"
                            width: parent.width
                            height: 30 * scaleFactor
                            font.bold: true
                            enabled: appManager.bestPredictedDigit >= 0
                            onClicked: appManager.learnLastDigit(correctDigitSelector.value)
                        }
                        Text {
                            text: "Model Evaluation"
                            color: "white"
                            font.pixelSize: Math.max(13, 14 * scaleFactor)
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
                            font.pixelSize: Math.max(9, Math.min(13, parent.height * 0.35))
                            Layout.fillWidth: true
                            horizontalAlignment: Text.AlignHCenter

                            wrapMode: isEvaluated? Text.NoWrap : Text.WordWrap
                            fontSizeMode: Text.Fit
                            minimumPixelSize:8
                        }
                    }
                    }
                }

                Rectangle {
                    id: programInfoCard
                    anchors.top: parent.top
                    anchors.topMargin: 12 * scaleFactor
                    anchors.left: teachCard.right
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
                            font.pixelSize: Math.max(13, 14 * scaleFactor)
                        }

                        Text {
                            width: parent.width
                            color: "#dddddd"
                            wrapMode: Text.WordWrap
                            font.pixelSize: Math.max(10, 11 * scaleFactor)

                            textFormat: Text.RichText
                            text: "This high-performance Qt/C++ app features a custom CNN built from scratch to recognize handwritten digits via MNIST. " +
                                  "It avoids external ML frameworks by pre-allocating memory buffers to eliminate heap overhead during training. " +
                                  "The architecture uses a convolutional layer with ReLU, max-pooling, flattening, and a 64-node hidden layer with softmax outputs. " +
                                  "To handle handwriting variations, it uses an augmentation pipeline (affine transformations, rotations, stroke dilation) " +
                                  "and accelerates training across multiple CPU threads via QtConcurrent. All this results in a 173,930 parameter model<br><br>" +
                                  "• <font color='#4da6ff'><b>Train Model:</b></font> Trains on MNIST data (~60,000 images).<br>" +
                                  "• <font color='#4da6ff'><b>Evaluate Model:</b></font> Tests accuracy on 10,000 test images.<br>" +
                                  "• <font color='#4da6ff'><b>Reset Model:</b></font> Reinitializes CNN with random weights.<br>" +
                                  "• <font color='#4da6ff'><b>Recognize Number:</b></font> Classifies your drawing and shows the top 3 matches.<br>" +
                                  "• <font color='#4da6ff'><b>Clear Number:</b></font> Resets the canvas."
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

            Text {
                text: appManager?.isTraining
                      ? "Training Progress: " + Math.round((appManager?.trainingProgress ?? 0) * 100) + "%"
                      : "Model is being evaluated..."
                color: "white"
                font.pixelSize: 22 * scaleFactor
                font.bold: true
                Layout.alignment: Qt.AlignHCenter
            }

            BusyIndicator {
                id: busyIndicator
                running: overlay.visible
                Layout.alignment: Qt.AlignHCenter
            }

            ProgressBar {
                id: trainingProgressBar
                Layout.fillWidth: true
                Layout.preferredHeight: 16 * scaleFactor
                visible: appManager.isTraining

                from: 0.0
                to: 1.0
                value: appManager?.trainingProgress ?? 0.0

                background: Rectangle {
                    implicitWidth: 200
                    implicitHeight: 16 * scaleFactor
                    color: "#2a2a2a"
                    radius: 8
                    border.color: "#444444"
                    border.width: 1
                }

                contentItem: Item {
                    implicitWidth: 200
                    implicitHeight: 16 * scaleFactor

                    Rectangle {
                        width: trainingProgressBar.visualPosition * parent.width
                        height: parent.height
                        radius: 8
                        color: "#12ff22"

                        Behavior on width {
                            NumberAnimation { duration: 150 }
                        }
                    }
                }
            }
        }
    }
}
