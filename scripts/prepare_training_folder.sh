#!/bin/bash

# Deplacement de tous les fichiers des sous repertoires dans dossier principal
dataDir='/home/leaf/tensorflow-for-poets-2/tf_files/flower_photos/'
if [ $dataDir = `` ] ; then
  echo "répertoire de donnees : pas de nom, fin du script"
else
  echo "repertoire de donnees : "$dataDir
  for plant in `ls $dataDir`
  do
    plantDir=$dataDir"/"$plant
    echo "repertoire fleur : "$plantDir
    for i in `ls $plantDir`
    do 
      if [ `ls -ld $plantDir/$i | cut -b1` = "d" ] ; then
        subDir=$plantDir/$i
        echo "sous repertoire : "$subDir
        ls $subDir
        mv $subDir/* $plantDir
        rmdir $subDir
      fi
    done
  done
fi

