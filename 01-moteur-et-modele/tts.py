import soundfile as sf, numpy as np, json
from kokoro_onnx import Kokoro
k = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
SEQ = [
 ["Pour comprendre un modèle économétrique de séries temporelles, faisons une analogie avec une voiture qui avance sur une route.",
  "Remarquez que Moteur et Modèle commencent tous les deux par la lettre M !",
  "Et pour fonctionner dans le temps, ils ont besoin d'un carburant qui commence par un D :",
  "du Diesel pour le moteur, et des Données historiques pour le modèle."],
 ["Quand on verse le Diesel, le moteur démarre et génère du mouvement.",
  "De la même façon, quand on estime le Modèle avec nos Données temporelles, on obtient des estimations de coefficients qui ont une véritable signification économique.",
  "Mais attention !",
  "Ce n'est pas parce que la voiture roule, ou que le logiciel donne un résultat, que tout fonctionne correctement."],
 ["Il faut faire passer des tests.",
  "D'abord, des tests mécaniques : on vérifie que toutes les pièces du moteur sont solidement fixées.",
  "En économétrie, cela revient à tester nos coefficients :",
  "on utilise les tests de Student et de Fisher pour s'assurer que nos variables sont statistiquement significatives, et bien à leur place."],
 ["Ensuite, il faut inspecter la fumée !",
  "Une fumée noire, c'est une combustion incomplète du diesel.",
  "Une fumée grasse, c'est une fuite d'huile.",
  "Il faut une fumée claire !",
  "Dans notre modèle, la fumée représente les Résidus.",
  "Pour valider le modèle, ces résidus doivent se comporter comme un Bruit Blanc :",
  "une espérance nulle,", "une variance constante,", "aucune autocorrélation, crucial dans le temps !", "et une loi normale."],
 ["Un moteur sain avec une fumée claire vous mènera loin.",
  "Un modèle dont les coefficients sont testés, et dont les résidus sont un bruit blanc, vous permettra de faire des prévisions économiques fiables.",
  "Bonne route en économétrie !"],
]
sr=24000; out=[]; timing=[]; t=0.0
lead=0.8
out.append(np.zeros(int(lead*sr),dtype=np.float32)); t=lead
for si,seq in enumerate(SEQ):
    st={"start":t,"lines":[]}
    for li,line in enumerate(seq):
        if "Student" in line or "Fisher" in line:
            ph=k.tokenizer.phonemize(line,lang="fr-fr")
            ph=ph.replace("stydˈɑ̃","stjˈuːdənt").replace("fiʃˈe","fˈɪʃə")
            print("PHONEMES:",ph)
            s,_=k.create(ph,voice="ff_siwis",speed=1.0,lang="fr-fr",is_phonemes=True)
        else:
            s,_=k.create(line,voice="ff_siwis",speed=1.0,lang="fr-fr")
        s=s.astype(np.float32)
        st["lines"].append([t,t+len(s)/sr,line]); out.append(s); t+=len(s)/sr
        gap=0.35 if li<len(seq)-1 else 0
        out.append(np.zeros(int(gap*sr),dtype=np.float32)); t+=gap
    tail=(3.2 if si==3 else 1.2) if si<len(SEQ)-1 else 3.0
    out.append(np.zeros(int(tail*sr),dtype=np.float32)); t+=tail
    st["end"]=t; timing.append(st)
a=np.concatenate(out); a=a/np.abs(a).max()*0.9
sf.write("voice.wav",a,sr)
json.dump(timing,open("timing.json","w"),ensure_ascii=False,indent=1)
for s in timing: print(round(s["start"],1),round(s["end"],1))
