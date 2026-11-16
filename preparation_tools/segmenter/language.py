"""Lightweight English/Spanish check from the transcript text (no extra dependency).

Counts common function words of each language, plus Spanish accents and ¿ ¡. Words shared by both
languages (a, no, me, he, son, come, ...) are left out of both lists. The result is "en", "es" or "mixed";
with too few cue words the configured default language is returned.
"""
import re

SPANISH = set("""
de la que el en y los se del las un por con una su para es al lo como pero sus le este porque esta entre
cuando muy sin sobre también hasta hay donde quien desde todo nos durante todos uno les ni contra otros ese
eso ante ellos esto antes algunos qué unos yo otro otras otra él tanto esa estos mucho quienes nada muchos
cual poco ella estar estas algunas algo nosotros mis tú ellas usted ustedes está están estoy tengo tiene
tienen quiero necesito puedo puede hola gracias señor señora bueno pues entonces ahora aquí sí también
cuenta dinero pago llamada factura tarjeta ayuda problema hablar español mañana hoy ayer después
nunca siempre cómo cuándo dónde por qué vez hace hacer dijo eres soy somos fue han había
""".split())
ENGLISH = set("""
the and to of i you it is that in my for this was on have with but be not are so what do can just we they
she at your it's i'm don't about would there if or all like will get know because yes okay ok um uh
been were has had did does could should their them then than when where which who why how
our us out up down here now very really well right thank thanks please sorry hello hi account money
payment call bill card help problem speak english today yesterday tomorrow never always going want need
""".split())
SPANISH_CHARS = re.compile(r"[áéíóúñü]")


def detect_language(text, spanish_share_min=0.8, english_share_max=0.2, min_cue_words=2, default="en"):
    """Return (language, spanish_cues, english_cues) for one transcript."""
    tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ']+", text.replace("’", "'"))
    # An accented word counts only in lower case: capitalised ones are often names (José, María).
    spanish = sum(1 for t in tokens if t.lower() in SPANISH or (t.islower() and SPANISH_CHARS.search(t)))
    spanish += text.count("¿") + text.count("¡")
    english = sum(1 for t in tokens if t.lower() in ENGLISH)
    if spanish + english < min_cue_words:
        return default, spanish, english
    share = spanish / (spanish + english)
    if share >= spanish_share_min:
        return "es", spanish, english
    if share <= english_share_max or spanish < min_cue_words:
        return "en", spanish, english
    if english < min_cue_words:
        return "es", spanish, english
    return "mixed", spanish, english
