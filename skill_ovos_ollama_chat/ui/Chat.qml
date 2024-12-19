import QtQuick 2.4
import QtQuick.Controls 2.2
import QtQuick.Layouts 1.4
import org.kde.kirigami 2.4 as Kirigami
import Mycroft 1.0 as Mycroft

Mycroft.Delegate {
    id: root
    property var rectColor: sessionData.fooColor

    onRectColorChanged: {
        fooRect.color = rectColor
    }
    Rectangle { // Title Row with percentage sizing using the Application Window as reference
        color: "#f8e6d9"
        anchors.centerIn: parentItem
        Label {
            text: "Chat"
            font.pointSize: 18 * (parentItem.width / applicationWindowWidth)
        }
    }
    Rectangle {
        id: fooRect
        anchors.fill: parent
        color: "#000"
        Text {
            id: text_field
            anchors.top: parent.top
            anchors.left: parent.left
            color: "#fff"
            height: parent.height
            width: parent.width
            text: sessionData.ChatText
            wrapMode: Text.WordWrap
            fontSizeMode: Text.Fit
            minimumPixelSize: 10
            font.pixelSize: 72
        }
    }
}
