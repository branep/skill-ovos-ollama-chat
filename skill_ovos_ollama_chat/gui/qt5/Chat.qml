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

    Rectangle {
        id: fooRect
        anchors.fill: parent
        color: "#000"

        ColumnLayout {
            anchors.fill: parent
            Label {
                id: chatText
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.margins: Mycroft.Units.gridUnit / 4
                text: sessionData.ChatText
                maximumLineCount: 1
                horizontalAlignment: Text.AlignHLeft
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
                minimumPixelSize: 5
                font.pixelSize: 42
                fontSizeMode: Text.Fit
                font.bold: false
                color: Kirigami.Theme.textColor
            }
        }
    }
}
