from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from peft import PeftModel
import torch

base_checkpoint = "facebook/mbart-large-50"
adapter_path = "./mbart-tuned-final"

tokenizer = MBart50TokenizerFast.from_pretrained(adapter_path)

base_model = MBartForConditionalGeneration.from_pretrained(base_checkpoint)

base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

base_model.generation_config.forced_bos_token_id = tokenizer.lang_code_to_id["de_DE"]

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def simplify_text(text: str, level: str = "Klexikon", max_new_tokens: int = 1024) -> str:
    #simplify text to either KLEXIKON or MINIKLEXIKON level 

    control_token = "<KLEXIKON>" if level == "Klexikon" else "<MINIKLEXIKON>"
    input_text = f"{control_token} {text}"

    tokenizer.src_lang = "de_DE"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"],
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

###TEST e.g.: ABBA###

artikel = """
ABBA ist eine schwedische Popgruppe, die aus den ehemaligen Paaren Agnetha Fältskog und Björn Ulvaeus sowie Benny Andersson und Anni-Frid Lyngstad besteht und sich 1972 in Stockholm formierte. Sie gehört mit rund 400 Millionen verkauften Tonträgern zu den erfolgreichsten Bands der Musikgeschichte.\nDie Gruppe hat die Geschichte der Popmusik wesentlich mitgeprägt. Vor ihnen hatte es keine Skandinavier mit vergleichbaren Erfolgen in der Popmusik gegeben. Trotz US-amerikanischer und britischer Dominanz im Musikgeschäft gelang ABBA der internationale Durchbruch. Nach ihrem Sieg beim Eurovision Song Contest 1974 mit Waterloo war sie besonders in Europa und Australien sowie später auch in Lateinamerika und Japan erfolgreich. Vor allem in der zweiten Hälfte der 1970er Jahre galt ABBA wegen der aufwendigen und ausgefeilten Musikproduktionen als Mitbegründer einer neuen internationalen Popmusik.\nZu ihren bekanntesten Songs zählen SOS, Mamma Mia, Fernando, Dancing Queen, Money, Money, Money, Thank You for the Music, Take a Chance on Me, Gimme! Gimme! Gimme! (A Man After Midnight), The Winner Takes It All, Lay All Your Love on Me und Super Trouper. Charakteristisch für die Gruppe waren außerdem ihre für die damalige Zeit ausgefallenen bunt poppigen Kostüme, die das Quartett bei Auftritten und in Musikvideos trug. Während Ulvaeus und Andersson hauptsächlich für Kompositionen und Texte verantwortlich waren, agierten Fältskog und Lyngstad vorwiegend als Leadsängerinnen.\nDas 1992 veröffentlichte Best-of-Album ABBA Gold gehört mit 31 Millionen verkauften Exemplaren zu den weltweit erfolgreichsten Alben und hat maßgeblich zum ABBA-Revival in den 1990ern beigetragen. 1999 hatte das auf ABBA-Songs basierende Musical Mamma Mia! in London Premiere, das mit über 60 Millionen Zuschauern eines der erfolgreichsten Musicals der Welt ist. Die Verfilmung von Mamma Mia! aus dem Jahr 2008 gehört darüber hinaus zu den erfolgreichsten Musicalverfilmungen.\n1982 beendeten die Gruppenmitglieder aufgrund privater Differenzen vorläufig ihre musikalische Zusammenarbeit und gaben erst 2018 bekannt, dass sie wieder zusammen an neuem Material arbeiteten. Obwohl es über einen Zeitraum von fast 40 Jahren weder Bandauftritte noch neue Musik gab, blieben ihre Lieder nach wie vor populär. So werden ihre Stücke weiterhin in unterschiedlichsten Interpretationen und Coverversionen produziert und meistens sehr erfolgreich vermarktet. Seit 2013 gibt es in Stockholm ein ABBA-Museum. 2021 gab die Band ihr Comeback im Rahmen der virtuellen Konzertshow ABBA Voyage bekannt und veröffentlichte vier neue Singles sowie ihr neuntes und finales Studioalbum Voyage.
"""

print("Klexikon (8-14 Jahre)")
print(simplify_text(artikel, level="Klexikon"))

print("\n MiniKlexikon (6-8 Jahre)")
print(simplify_text(artikel, level="MiniKlexikon"))