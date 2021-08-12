#!/usr/bin/env bash

set -eux

if [ ! -f linuxdeploy-x86_64.AppImage ]
then
  curl -L -O https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
fi
if [ ! -f linuxdeploy-plugin-appimage-x86_64.AppImage ]
then
  curl -L -O https://github.com/linuxdeploy/linuxdeploy-plugin-appimage/releases/download/continuous/linuxdeploy-plugin-appimage-x86_64.AppImage
fi

chmod +x linuxdeploy-*.AppImage
rm -rf appimage/appdir
mkdir -p appimage/appdir/usr/bin
cp $1 appimage/appdir/usr/bin/

bn=`basename $1`

echo "[Desktop Entry]
Name=$bn
Exec=$bn
Icon=$bn
Type=Application
Terminal=true
Categories=Development;" > appimage/$bn.desktop

curl -L -o appimage/$bn.png https://gist.github.com/daquexian/cd140d70b1772daa4a736c0642a68e9d/raw/da2b026ac950623700ddfb199d987c55215fc9cb/white-icon.png

./linuxdeploy-x86_64.AppImage --appdir appimage/appdir -d appimage/$bn.desktop -i appimage/$bn.png --output appimage

rm -rf appimage/appdir
rm -rf linuxdeploy-*.AppImage
