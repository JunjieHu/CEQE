


link="https://www.quest.dcs.shef.ac.uk/wmt18_files_qe/features_"

for langs in en_de de_en en_cs en_lv; do
    wget "${link}${langs}_test.tar.gz"
    wget "$link$langs.tar.gz"
    tar -zxvf features_${langs}_test.tar.gz
    tar -zxvf features_${langs}.tar.gz
done

mv from_varvara/features .
mv en_cs features
mv en_lv features
mv en_de features 
rm -rf from_varvara # features*.tar.gz
