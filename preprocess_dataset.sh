DIR=$1
OUTPUT_DIR=$2

for speakerdir in ${DIR}/*/
do
    for chapterdir in $speakerdir/*/
    do
        mv ${chapterdir}/*.txt $OUTPUT_DIR
        for flacfile in ${chapterdir}/*.flac
        do
            filename=$(basename $flacfile .flac)
            ffmpeg -i $flacfile ${OUTPUT_DIR}/${filename}.wav
        done
    done
done
