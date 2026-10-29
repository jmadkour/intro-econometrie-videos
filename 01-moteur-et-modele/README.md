# L'analogie du Moteur et du Modèle économétrique (séries temporelles)

Vidéo pédagogique (1 min 49, 1280×720) destinée aux étudiants de Licence 3 de Sciences Économiques.
Elle compare un moteur de voiture à un modèle économétrique de séries temporelles, en cinq séquences :

1. **M & D et le temps** : Moteur / Modèle, Diesel / Données historiques
2. **Démarrage et estimation** : le diesel fait démarrer le moteur, les données donnent les coefficients estimés
3. **Tests mécaniques vs tests sur les coefficients** : tests t de Student et F de Fisher
4. **La fumée vs les résidus** : espérance nulle, homoscédasticité, absence d'autocorrélation, normalité, donc **bruit blanc**
5. **Conclusion** : un modèle validé permet des prévisions fiables

La vidéo finale est `moteur_modele_econometrique.mp4`. Le scénario d'origine est dans `scenario.txt`.

> Les valeurs numériques affichées (β̂₀ = 2,14, β̂₁ = 0,71, t, F) sont illustratives.

## Régénérer la vidéo

Prérequis : Python 3.10+, `ffmpeg`, les polices DejaVu.

```bash
pip install -r requirements.txt

# Modèle de synthèse vocale Kokoro (≈ 350 Mo, non versionné)
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

python tts.py                                   # voix off -> voice.wav + timing.json
python anim.py video moteur_modele_econometrique.mp4   # animation + montage
python anim.py still 60 90.6                    # (option) images fixes pour vérifier
```

- `tts.py` contient le texte de la voix off, séquence par séquence. « Student » et « Fisher » y sont prononcés à l'anglaise grâce à une transcription phonétique.
- `anim.py` dessine chaque image (Pillow + formules matplotlib) en se calant sur `timing.json`, puis assemble le tout avec ffmpeg.

## Auteur

Jaouad Madkour, Faculté d'Économie et de Gestion de Tanger, Université Abdelmalek Essaâdi.
Cours : [jmadkour.org](https://jmadkour.org) · YouTube : [@JaouadMadkour](https://www.youtube.com/@JaouadMadkour)
