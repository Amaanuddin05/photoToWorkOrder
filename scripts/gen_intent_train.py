"""Generate eval/intent_train.csv: ~1,000 labelled EN+AR radio-style messages (issue / done / other).

Composed from hand-written topic phrases x locations x style modifiers. Split is by TOPIC (dev topics never
appear in train) so dev accuracy measures generalisation, not memorisation.  Reproducible (seed 7).
Usage: ..\\venv\\Scripts\\python.exe scripts\\gen_intent_train.py
"""
import csv, random
from pathlib import Path

random.seed(7)
OUT = Path(__file__).resolve().parent.parent / "eval" / "intent_train.csv"
PER_CLASS_LANG = 180

LOC = {
    "en": ["room 204", "the 2nd floor washroom", "block B", "the library", "the main corridor", "the canteen",
           "the staff room", "reception", "the workshop", "the car park", "level 3", "the gym", "classroom 5",
           "the store room", "the back gate", "the meeting room", "the pump room", "the front entrance",
           "building A", "the kitchen"],
    "ar": ["الغرفة 204", "دورة المياه في الطابق الثاني", "المبنى ب", "المكتبة", "الممر الرئيسي", "المقصف",
           "غرفة الموظفين", "الاستقبال", "الورشة", "مواقف السيارات", "الطابق الثالث", "النادي الرياضي",
           "الفصل 5", "المخزن", "البوابة الخلفية", "غرفة الاجتماعات", "غرفة المضخات", "المدخل الأمامي",
           "المبنى أ", "المطبخ"],
}

# topic: (EN issue x3, EN done x3, AR issue x3, AR done x3)   {l} = location
T = {
 "ac": (["AC in {l} is not cooling", "The air conditioner in {l} is blowing warm air", "AC unit dripping water in {l}"],
        ["AC in {l} fixed and cooling well now", "Refilled the gas for the AC in {l} job done", "Cleaned the AC filters in {l} completed"],
        ["المكيف في {l} لا يبرد", "المكيف في {l} يخرج هواء ساخن", "تتساقط المياه من المكيف في {l}"],
        ["تم إصلاح المكيف في {l} ويبرد بشكل جيد", "تمت تعبئة غاز المكيف في {l} وانتهى العمل", "تم تنظيف فلاتر المكيف في {l}"]),
 "tap": (["Tap in {l} keeps leaking", "Water dripping from the faucet in {l}", "The mixer tap in {l} is broken and leaking"],
         ["Fixed the leaking tap in {l}", "Replaced the washer on the tap in {l} all good", "Tap in {l} repaired"],
         ["الصنبور في {l} يسرب الماء باستمرار", "تقطر المياه من الحنفية في {l}", "خلاط المياه في {l} مكسور ويسرب"],
         ["تم إصلاح الصنبور المسرب في {l}", "تم تغيير جلدة الحنفية في {l} وكل شيء تمام", "تم إصلاح الحنفية في {l}"]),
 "toilet": (["Toilet in {l} is blocked and overflowing", "The flush is not working in {l}", "Toilet seat broken in {l}"],
            ["Toilet blockage cleared in {l}", "Flush mechanism replaced in {l} working now", "Toilet in {l} repaired"],
            ["المرحاض في {l} مسدود ويفيض", "السيفون لا يعمل في {l}", "مقعد المرحاض مكسور في {l}"],
            ["تم فتح انسداد المرحاض في {l}", "تم تبديل السيفون في {l} ويعمل الآن", "تم إصلاح المرحاض في {l}"]),
 "drain": (["Bad smell from the drain in {l}", "Drain in {l} is blocked, water not going down", "Sewage smell near {l}"],
           ["Drain in {l} cleaned and the smell is gone", "Unblocked the drain in {l} done", "Sewer line near {l} flushed all clear"],
           ["رائحة كريهة من المجاري في {l}", "المصرف في {l} مسدود والماء لا ينزل", "رائحة صرف صحي قرب {l}"],
           ["تم تنظيف المصرف في {l} وزالت الرائحة", "تم فتح المصرف المسدود في {l} تم", "تم غسل خط الصرف قرب {l}"]),
 "lights": (["Lights are flickering in {l}", "Tube light not working in {l}", "Two bulbs are out in {l}"],
            ["Replaced the tube lights in {l} done", "Lighting in {l} fixed", "Changed the bulbs in {l} all working"],
            ["الإضاءة تومض في {l}", "مصباح لا يعمل في {l}", "لمبتان محروقتان في {l}"],
            ["تم تغيير الإضاءة في {l} وانتهى", "تم إصلاح الإضاءة في {l}", "تم تبديل اللمبات في {l} وتعمل جميعها"]),
 "socket": (["Power socket in {l} is sparking", "No electricity at the sockets in {l}", "Switch in {l} is hot and burning smell"],
            ["Socket in {l} replaced and tested", "Power restored in {l} breaker fixed", "Changed the faulty switch in {l} all fine"],
            ["المقبس الكهربائي في {l} يطلق شرارة", "لا توجد كهرباء في المقابس في {l}", "المفتاح في {l} ساخن وفيه رائحة حرق"],
            ["تم استبدال المقبس في {l} واختباره", "عادت الكهرباء في {l} بعد إصلاح القاطع", "تم تغيير المفتاح التالف في {l} وكل شيء سليم"]),
 "lift": (["Lift near {l} is stuck", "Elevator not working at {l}", "Lift doors will not close at {l}"],
          ["Lift at {l} is back in service", "Technician finished the lift at {l} running fine", "Lift door at {l} adjusted and working"],
          ["المصعد قرب {l} عالق", "المصعد لا يعمل عند {l}", "أبواب المصعد لا تغلق عند {l}"],
          ["المصعد عند {l} عاد للخدمة", "انتهى الفني من المصعد عند {l} ويعمل جيدًا", "تم ضبط باب المصعد عند {l} ويعمل"]),
 "lock": (["Door lock in {l} is broken", "Cannot lock the door of {l}, key stuck", "Door handle fell off in {l}"],
          ["Changed the lock in {l} done", "Door lock in {l} repaired", "Fitted a new handle on the door of {l}"],
          ["قفل الباب في {l} مكسور", "لا أستطيع قفل باب {l} والمفتاح عالق", "سقط مقبض الباب في {l}"],
          ["تم تغيير القفل في {l}", "تم إصلاح قفل الباب في {l}", "تم تركيب مقبض جديد لباب {l}"]),
 "window": (["Window glass cracked in {l}", "Window in {l} will not close", "Broken window in {l} glass on the floor"],
            ["New window glass fitted in {l}", "Window latch fixed in {l}", "Broken glass replaced in {l} and cleaned"],
            ["زجاج النافذة متشقق في {l}", "النافذة في {l} لا تغلق", "نافذة مكسورة في {l} والزجاج على الأرض"],
            ["تم تركيب زجاج جديد للنافذة في {l}", "تم إصلاح مزلاج النافذة في {l}", "تم استبدال الزجاج المكسور في {l} وتنظيفه"]),
 "ceiling": (["Ceiling tile fell in {l}", "Ceiling in {l} has a big stain and sagging", "Plaster coming off the ceiling in {l}"],
             ["Ceiling tile put back in {l}", "Ceiling repaired and painted in {l}", "Fixed the false ceiling in {l}"],
             ["سقطت قطعة من السقف في {l}", "السقف في {l} فيه بقعة كبيرة ومتهدل", "الجبس يتساقط من السقف في {l}"],
             ["تمت إعادة تركيب قطعة السقف في {l}", "تم إصلاح السقف ودهانه في {l}", "تم إصلاح السقف المستعار في {l}"]),
 "roof": (["Roof leaking when it rains in {l}", "Water coming through the roof above {l}", "Damp patch on the wall in {l}"],
          ["Roof leak above {l} patched", "Waterproofing done on the roof of {l}", "Damp wall in {l} treated and painted"],
          ["السقف يتسرب عند المطر في {l}", "الماء ينزل من السقف فوق {l}", "بقعة رطوبة على الجدار في {l}"],
          ["تم ترقيع تسرب السقف فوق {l}", "تم عزل سقف {l} ضد الماء", "تمت معالجة رطوبة الجدار في {l} ودهانه"]),
 "pipe": (["Pipe burst near {l} water everywhere", "Water pipe leaking in {l}", "Big water leak in the ceiling of {l}"],
          ["Burst pipe near {l} repaired", "Water pipe in {l} fixed no more leak", "Isolated and fixed the leaking pipe in {l}"],
          ["انفجر أنبوب قرب {l} والماء في كل مكان", "أنبوب المياه يسرب في {l}", "تسرب كبير للماء في سقف {l}"],
          ["تم إصلاح الأنبوب المنفجر قرب {l}", "تم إصلاح أنبوب المياه في {l} ولا يوجد تسرب", "تم عزل وإصلاح الأنبوب المسرب في {l}"]),
 "fire_alarm": (["Fire alarm panel showing fault in {l}", "Smoke detector beeping in {l}", "Fire alarm going off for no reason in {l}"],
                ["Fire alarm panel reset in {l} fault cleared", "Smoke detector battery changed in {l}", "Fire alarm in {l} tested and OK"],
                ["لوحة إنذار الحريق تظهر عطل في {l}", "كاشف الدخان يصفر في {l}", "إنذار الحريق يعمل بدون سبب في {l}"],
                ["تمت إعادة ضبط لوحة الإنذار في {l} وزال العطل", "تم تغيير بطارية كاشف الدخان في {l}", "تم اختبار إنذار الحريق في {l} وهو سليم"]),
 "generator": (["Generator is not starting", "Generator making strange noise and smoking", "Backup power did not switch on in {l}"],
               ["Generator serviced and running normally", "Generator fuel filled and started fine", "Backup power restored in {l}"],
               ["المولد لا يشتغل", "المولد يصدر صوتًا غريبًا ويدخن", "الكهرباء الاحتياطية لم تعمل في {l}"],
               ["تمت صيانة المولد ويعمل بشكل طبيعي", "تمت تعبئة وقود المولد وبدأ يعمل", "عادت الكهرباء الاحتياطية في {l}"]),
 "gate": (["Gate motor not working at {l}", "Barrier gate stuck open at {l}", "Sliding gate at {l} jammed"],
          ["Gate motor at {l} repaired", "Barrier at {l} fixed and moving", "Sliding gate at {l} unjammed and lubricated"],
          ["محرك البوابة لا يعمل عند {l}", "حاجز البوابة عالق مفتوح عند {l}", "البوابة المنزلقة عند {l} عالقة"],
          ["تم إصلاح محرك البوابة عند {l}", "تم إصلاح الحاجز عند {l} ويتحرك", "تم فك البوابة المنزلقة عند {l} وتشحيمها"]),
 "cctv": (["CCTV camera at {l} is offline", "Camera in {l} showing black screen", "Camera pointing the wrong way at {l}"],
          ["CCTV at {l} back online", "Camera in {l} replaced and working", "Camera at {l} re-aimed"],
          ["كاميرا المراقبة عند {l} متوقفة", "الكاميرا في {l} تعرض شاشة سوداء", "الكاميرا عند {l} متجهة بالاتجاه الخطأ"],
          ["كاميرا المراقبة عند {l} عادت للعمل", "تم استبدال الكاميرا في {l} وتعمل", "تم تعديل اتجاه الكاميرا عند {l}"]),
 "cooler": (["Water cooler in {l} is not working", "Drinking water dispenser in {l} has no cold water", "Water cooler in {l} leaking"],
            ["Water cooler in {l} fixed", "Dispenser in {l} serviced cold water back", "Replaced the water cooler filter in {l}"],
            ["برادة المياه في {l} لا تعمل", "موزع المياه في {l} لا يبرد", "برادة المياه في {l} تسرب"],
            ["تم إصلاح برادة المياه في {l}", "تمت صيانة الموزع في {l} وعاد الماء البارد", "تم تغيير فلتر البرادة في {l}"]),
 "gas": (["Smell of gas in {l}", "Gas leak near the cooker in {l}", "Gas cylinder hissing in {l}"],
         ["Gas leak in {l} sealed area checked and safe", "Gas pipe in {l} replaced and tested", "Cylinder valve in {l} fixed"],
         ["رائحة غاز في {l}", "تسرب غاز قرب الموقد في {l}", "أسطوانة الغاز تصفر في {l}"],
         ["تم إغلاق تسرب الغاز في {l} والمنطقة آمنة", "تم تغيير أنبوب الغاز في {l} واختباره", "تم إصلاح صمام الأسطوانة في {l}"]),
 "sink": (["Sink in {l} is blocked", "Basin in {l} not draining", "Sink in {l} is cracked and leaking"],
          ["Sink in {l} unblocked", "Basin drain in {l} cleared", "Replaced the sink in {l}"],
          ["المغسلة في {l} مسدودة", "الحوض في {l} لا يصرف الماء", "المغسلة في {l} متشققة وتسرب"],
          ["تم فتح انسداد المغسلة في {l}", "تم تسليك مصرف الحوض في {l}", "تم استبدال المغسلة في {l}"]),
 "handrail": (["Handrail on the stairs near {l} is loose", "Stair railing broken at {l}", "Step is cracked on the stairs at {l}"],
              ["Handrail near {l} tightened and secured", "Railing at {l} welded and fixed", "Cracked step at {l} repaired"],
              ["الدرابزين على الدرج قرب {l} مرتخٍ", "درابزين الدرج مكسور عند {l}", "درجة الدرج متشققة عند {l}"],
              ["تم شد الدرابزين قرب {l} وتثبيته", "تم لحام وإصلاح الدرابزين عند {l}", "تم إصلاح الدرجة المتشققة عند {l}"]),
 "heater": (["No hot water in {l}", "Water heater in {l} is not heating", "Geyser in {l} is leaking"],
            ["Water heater in {l} repaired hot water is back", "Replaced the heating element in {l}", "Geyser in {l} fixed"],
            ["لا يوجد ماء ساخن في {l}", "السخان في {l} لا يسخن", "السخان في {l} يسرب"],
            ["تم إصلاح السخان في {l} وعاد الماء الساخن", "تم تغيير عنصر التسخين في {l}", "تم إصلاح السخان في {l}"]),
 "exit_light": (["Emergency exit light is not working in {l}", "Exit sign in {l} is dark", "Emergency light in {l} is flickering"],
                ["Emergency exit light in {l} replaced", "Exit sign in {l} fixed and lit", "Emergency lights in {l} tested all OK"],
                ["إضاءة مخرج الطوارئ لا تعمل في {l}", "لوحة المخرج في {l} مظلمة", "إضاءة الطوارئ في {l} تومض"],
                ["تم تبديل إضاءة مخرج الطوارئ في {l}", "تم إصلاح لوحة المخرج في {l} وتضيء", "تم اختبار إضاءة الطوارئ في {l} وكلها سليمة"]),
 "bins": (["Garbage bins overflowing near {l}", "Rats seen near {l} we need pest control", "Cockroaches in {l}"],
          ["Bins near {l} emptied and cleaned", "Pest control done in {l}", "Sprayed for cockroaches in {l} done"],
          ["حاويات القمامة ممتلئة قرب {l}", "شوهدت فئران قرب {l} نحتاج مكافحة آفات", "صراصير في {l}"],
          ["تم تفريغ الحاويات قرب {l} وتنظيفها", "تمت مكافحة الآفات في {l}", "تم رش المبيد في {l} وانتهى"]),
 "tiles": (["Floor tiles broken at {l} people may trip", "Loose tiles in {l}", "Floor in {l} is cracked and uneven"],
           ["Tiles at {l} replaced", "Loose tiles in {l} fixed", "Floor in {l} repaired and levelled"],
           ["بلاط الأرضية مكسور عند {l} وقد يتعثر الناس", "بلاط مفكوك في {l}", "الأرضية في {l} متشققة وغير مستوية"],
           ["تم استبدال البلاط عند {l}", "تم تثبيت البلاط المفكوك في {l}", "تم إصلاح الأرضية في {l} وتسويتها"]),
}

T.update({
 "facade": (["The outer wall of {l} is covered in black streaks and dirt", "Algae growing on the external wall near {l}", "Building facade at {l} needs washing, very dirty"],
            ["Facade near {l} washed and cleaned", "Cleaned the external wall at {l} all the stains are gone", "Pressure washed the outside wall of {l} done"],
            ["الجدار الخارجي عند {l} مغطى بالأوساخ والخطوط السوداء", "طحالب تنمو على الجدار الخارجي قرب {l}", "واجهة المبنى عند {l} تحتاج إلى غسيل وهي متسخة جدًا"],
            ["تم غسل واجهة المبنى قرب {l} وتنظيفها", "تم تنظيف الجدار الخارجي عند {l} وزالت البقع", "تم غسل الجدار الخارجي عند {l} بالضغط وانتهى"]),
 "locker": (["Locker in {l} will not open, key turns but nothing happens", "Teacher locked out of {l}, key stuck in the lock", "I am locked out of {l}, need someone with a key"],
            ["Opened the locker in {l} for the student", "Got the teacher back into {l} lock freed", "Unlocked {l} and checked the key works now"],
            ["الخزانة في {l} لا تفتح والمفتاح يدور دون فائدة", "المعلم عالق خارج {l} والمفتاح عالق في القفل", "أنا مقفل خارج {l} أحتاج شخصًا معه مفتاح"],
            ["تم فتح الخزانة في {l} للطالب", "تم إدخال المعلم إلى {l} وتحرير القفل", "تم فتح {l} والتأكد أن المفتاح يعمل الآن"]),
 "spill": (["Someone threw up in {l} needs urgent cleaning", "Big spill on the floor of {l} please clean now", "Broken bottle and liquid on the floor in {l}"],
           ["Spill in {l} cleaned and mopped", "Vomit in {l} cleaned and disinfected", "Cleaned up the mess in {l} floor is dry"],
           ["أحدهم تقيأ في {l} ويلزم تنظيف عاجل", "انسكاب كبير على أرضية {l} يرجى التنظيف الآن", "زجاجة مكسورة وسائل على الأرض في {l}"],
           ["تم تنظيف الانسكاب في {l} ومسح الأرض", "تم تنظيف وتعقيم القيء في {l}", "تم تنظيف الفوضى في {l} والأرض جافة"]),
 "secure_area": (["Fallen tree blocking the path near {l} secure the area", "Sparking panel in {l} keep people away", "Danger: broken glass everywhere in {l} cordon it off"],
                 ["Area near {l} secured and cordoned", "Tree removed from the path near {l}", "Made the area at {l} safe and cleared people"],
                 ["شجرة ساقطة تسد الطريق قرب {l} يرجى تأمين المنطقة", "لوحة كهرباء تطلق شرارًا في {l} أبعدوا الناس", "خطر: زجاج مكسور في كل مكان في {l} يرجى عزل المنطقة"],
                 ["تم تأمين المنطقة قرب {l} وتطويقها", "تمت إزالة الشجرة عن الطريق قرب {l}", "تم جعل المنطقة عند {l} آمنة وإخلاء الناس"]),
 "ext_clean": (["Fallen leaves all over the walkway near {l}", "The area around {l} is dirty and needs sweeping", "Litter spread across the yard at {l} after the weekend"],
               ["Swept the walkway near {l}", "Cleaned up the outside area around {l}", "Collected the litter in the yard at {l} done"],
               ["أوراق الشجر متساقطة على الممشى قرب {l}", "المنطقة حول {l} متسخة وتحتاج إلى كنس", "نفايات منتشرة في الساحة عند {l} بعد عطلة الأسبوع"],
               ["تم كنس الممشى قرب {l}", "تم تنظيف المنطقة الخارجية حول {l}", "تم جمع النفايات في الساحة عند {l} وانتهى"]),
 "move_furn": (["Please move 20 desks from {l} to the other block", "Shift the sofa in {l} to reception", "Need the cupboards in {l} moved to the new office"],
               ["Moved the desks from {l} as requested", "Shifted the sofa out of {l} done", "Cupboards in {l} relocated to the new office"],
               ["يرجى نقل 20 طاولة من {l} إلى المبنى الآخر", "انقلوا الأريكة في {l} إلى الاستقبال", "نحتاج إلى نقل الخزائن في {l} إلى المكتب الجديد"],
               ["تم نقل الطاولات من {l} كما طلبتم", "تم نقل الأريكة من {l} وانتهى", "تم نقل الخزائن في {l} إلى المكتب الجديد"]),
 "gen_clean": (["{l} floor is dusty and needs mopping", "Dirty corridor near {l} needs sweeping", "Toilets in {l} are dirty please clean"],
               ["Mopped the floor of {l} done", "Swept and cleaned the corridor near {l}", "Cleaned the toilets in {l} completed"],
               ["أرضية {l} مغبرة وتحتاج إلى مسح", "الممر قرب {l} متسخ ويحتاج إلى كنس", "دورات المياه في {l} متسخة يرجى تنظيفها"],
               ["تم مسح أرضية {l} وانتهى", "تم كنس وتنظيف الممر قرب {l}", "تم تنظيف دورات المياه في {l} وانتهى"]),
 "key_copy": (["Need a copy of the key for {l}", "Please make a spare key for the cabinet in {l}", "New staff member needs a duplicate key for {l}"],
              ["Made two copies of the key for {l}", "Spare key for {l} cut and handed over", "Duplicate key for {l} ready and delivered"],
              ["نحتاج نسخة من مفتاح {l}", "يرجى عمل مفتاح احتياطي للخزانة في {l}", "الموظف الجديد يحتاج إلى نسخة مفتاح لـ {l}"],
              ["تم عمل نسختين من مفتاح {l}", "تم قص المفتاح الاحتياطي لـ {l} وتسليمه", "نسخة المفتاح لـ {l} جاهزة وتم تسليمها"]),
 "office_clean": (["Vacuum the carpet in {l}", "Clean the desks and monitors in {l}", "Admin offices in {l} need dusting and cleaning"],
                  ["Vacuumed the carpet in {l}", "Cleaned the desks and monitors in {l} done", "Dusted and cleaned the offices in {l}"],
                  ["نظفوا السجاد بالمكنسة الكهربائية في {l}", "نظفوا المكاتب والشاشات في {l}", "مكاتب الإدارة في {l} تحتاج إلى تنظيف وإزالة الغبار"],
                  ["تم تنظيف السجاد بالمكنسة في {l}", "تم تنظيف المكاتب والشاشات في {l} وانتهى", "تم تنظيف المكاتب في {l} وإزالة الغبار"]),
 "roof_clean": (["Dead leaves and debris on the roof above {l}", "Water collecting on the terrace of {l} with dirt", "Gutters on the roof of {l} are full of leaves"],
                ["Roof above {l} cleared of debris", "Cleaned the terrace of {l} and drains are flowing", "Gutters on the roof of {l} emptied"],
                ["أوراق ميتة وحطام على السطح فوق {l}", "الماء يتجمع على سطح {l} مع الأوساخ", "المزاريب على سطح {l} ممتلئة بالأوراق"],
                ["تم تنظيف السطح فوق {l} من الحطام", "تم تنظيف سطح {l} والمصارف تعمل", "تم تفريغ المزاريب على سطح {l}"]),
 "patrol": (["Security guard was not at the post near {l} during the night round", "Patrol skipped {l} last night", "Gate at {l} found unlocked after hours no guard"],
            ["Patrol of {l} completed and secured", "Guard back at the post near {l}", "Gate at {l} locked and checked"],
            ["حارس الأمن لم يكن في موقعه قرب {l} أثناء الجولة الليلية", "تم تجاوز {l} في الجولة أمس", "بوابة {l} وجدت مفتوحة بعد الدوام ولا يوجد حارس"],
            ["تمت جولة التفتيش في {l} وتأمينه", "عاد الحارس إلى موقعه قرب {l}", "تم قفل بوابة {l} والتأكد منها"]),
 "event_setup": (["Set up the stage and chairs in {l} for annual day", "Set up the podium and microphone in {l} for assembly", "Need tables and tents arranged at {l} for the event"],
                 ["Stage and chairs in {l} set up and ready", "Podium and mic in {l} installed", "Tents and tables at {l} arranged as requested"],
                 ["جهزوا المسرح والكراسي في {l} لليوم السنوي", "جهزوا المنصة والميكروفون في {l} للطابور", "نحتاج ترتيب الطاولات والخيام عند {l} للفعالية"],
                 ["تم تجهيز المسرح والكراسي في {l} وهو جاهز", "تم تركيب المنصة والميكروفون في {l}", "تم ترتيب الخيام والطاولات عند {l} كما طُلب"]),
 "volleyball": (["Put up the volleyball net in {l}", "Net is sagging in {l} needs tightening before the match", "Volleyball court at {l} needs the lines and net set"],
                ["Volleyball net put up in {l}", "Net in {l} tightened and ready for the match", "Court at {l} marked and net fitted"],
                ["ركّبوا شبكة الكرة الطائرة في {l}", "الشبكة متهدلة في {l} وتحتاج إلى شد قبل المباراة", "ملعب الكرة الطائرة عند {l} يحتاج إلى الخطوط والشبكة"],
                ["تم تركيب شبكة الكرة الطائرة في {l}", "تم شد الشبكة في {l} وهي جاهزة للمباراة", "تم تخطيط الملعب عند {l} وتركيب الشبكة"]),
 "water_bottle": (["Water dispenser in {l} is empty needs a new bottle", "Empty water jug at the station in {l}", "Drinking station near {l} has no water bottle left"],
                  ["Replaced the water bottle at {l}", "New jug fitted at the dispenser in {l}", "Refilled the drinking stations near {l}"],
                  ["موزع المياه في {l} فارغ ويحتاج إلى قارورة جديدة", "جالون الماء فارغ عند المحطة في {l}", "محطة الشرب قرب {l} لا يوجد بها قارورة"],
                  ["تم استبدال قارورة الماء عند {l}", "تم تركيب جالون جديد في الموزع في {l}", "تمت تعبئة محطات الشرب قرب {l}"]),
 "deep_clean": (["Washroom tiles in {l} are stained and smell bad needs scrubbing", "Limescale and stains in {l} that normal cleaning cannot remove", "Toilets in {l} need a deep clean"],
                ["Deep cleaned the washroom in {l}", "Scrubbed the tiles and removed the limescale in {l}", "Deep cleaning of {l} finished"],
                ["بلاط دورة المياه في {l} متسخ وله رائحة ويحتاج إلى فرك", "ترسبات كلسية وبقع في {l} لا يزيلها التنظيف العادي", "دورات المياه في {l} تحتاج إلى تنظيف عميق"],
                ["تم التنظيف العميق لدورة المياه في {l}", "تم فرك البلاط وإزالة الترسبات في {l}", "انتهى التنظيف العميق في {l}"]),
 "waste": (["Dustbins near {l} are full and need emptying", "Garbage overflowing behind {l}", "Skip bin at {l} has not been collected for days"],
           ["Dustbins near {l} emptied", "Garbage behind {l} collected", "Skip bin at {l} cleared today"],
           ["صناديق القمامة قرب {l} ممتلئة وتحتاج إلى تفريغ", "القمامة تفيض خلف {l}", "حاوية النفايات عند {l} لم تُجمع منذ أيام"],
           ["تم تفريغ صناديق القمامة قرب {l}", "تم جمع القمامة خلف {l}", "تم إفراغ حاوية النفايات عند {l} اليوم"]),
 "window_clean": (["Windows in {l} are dirty and streaky", "Clean the glass windows in {l} inside and outside", "Exterior glass at {l} needs cleaning"],
                  ["Cleaned the windows in {l} inside and out", "Glass at {l} washed and streak free", "Exterior windows at {l} cleaned"],
                  ["النوافذ في {l} متسخة وعليها خطوط", "نظفوا زجاج النوافذ في {l} من الداخل والخارج", "الزجاج الخارجي عند {l} يحتاج إلى تنظيف"],
                  ["تم تنظيف النوافذ في {l} من الداخل والخارج", "تم غسل الزجاج عند {l} وأصبح نظيفًا", "تم تنظيف النوافذ الخارجية عند {l}"]),
 "handyman": (["Door handle came off in {l}", "Hinge on the cabinet door in {l} is broken", "Shelf in {l} is hanging loose from the wall"],
              ["Door handle in {l} fixed back on", "Replaced the hinge on the cabinet in {l}", "Shelf in {l} fixed to the wall"],
              ["مقبض الباب انفصل في {l}", "مفصلة باب الخزانة في {l} مكسورة", "الرف في {l} متدلٍّ عن الجدار"],
              ["تم تركيب مقبض الباب في {l} من جديد", "تم تبديل مفصلة الخزانة في {l}", "تم تثبيت الرف في {l} بالجدار"]),
 "it_equip": (["Projector in {l} won't turn on", "Wi-Fi is not working in {l}", "Printer in {l} is jammed and not printing"],
              ["Projector in {l} fixed and working", "Wi-Fi restored in {l}", "Printer in {l} unjammed and printing again"],
              ["جهاز العرض في {l} لا يشتغل", "الواي فاي لا يعمل في {l}", "الطابعة في {l} عالقة ولا تطبع"],
              ["تم إصلاح جهاز العرض في {l} ويعمل", "عاد الواي فاي في {l}", "تم فك الطابعة في {l} وتطبع من جديد"]),
})

# EN/AR "mixed / tricky" messages (topic "mixed"): label -> list
MIXED = {
 "issue": {"en": ["Fixed the AC in room 204 but now the lift is stuck", "Thanks for the tap repair, the other tap in the same washroom is also leaking",
                  "Cleaned the area but the ceiling is still leaking please recheck", "Reported yesterday and still not repaired, the light is out in the corridor",
                  "Painters left but the drain is blocked now", "Replaced the bulb but it flickers again",
                  "The repair did not work the toilet is leaking again", "Job was marked done but the socket is still dead",
                  "Everything fine except the gate motor which stopped", "Done with level 2, but level 3 AC is not cooling"],
           "ar": ["تم إصلاح المكيف في الغرفة 204 لكن المصعد عالق الآن", "شكرًا على إصلاح الصنبور لكن الصنبور الآخر في نفس الحمام يسرب",
                  "تم التنظيف لكن السقف ما زال يتسرب يرجى إعادة الفحص", "تم الإبلاغ أمس ولم يتم الإصلاح بعد الإضاءة معطلة في الممر",
                  "غادر الدهانون لكن المصرف مسدود الآن", "تم تغيير اللمبة لكنها تومض مرة أخرى",
                  "الإصلاح لم ينجح المرحاض يسرب مرة أخرى", "تم تسجيل العمل كمنتهٍ لكن المقبس ما زال لا يعمل",
                  "كل شيء تمام ما عدا محرك البوابة توقف", "انتهينا من الطابق الثاني لكن مكيف الطابق الثالث لا يبرد"]},
 "done": {"en": ["All sorted now, thanks", "Completed as requested", "Work finished, please close the ticket", "Done and tested, all working",
                 "Issue resolved no further action needed", "Job complete, site cleaned", "Task finished at 4 pm",
                 "Repaired, you can use it now", "Everything is back to normal", "Finished the job, photo attached"],
          "ar": ["تم الانتهاء شكرًا", "تم التنفيذ كما هو مطلوب", "انتهى العمل يرجى إغلاق البلاغ", "تم وتم اختباره ويعمل بشكل سليم",
                 "تم حل المشكلة ولا حاجة لأي إجراء", "اكتمل العمل وتم تنظيف الموقع", "انتهت المهمة الساعة الرابعة",
                 "تم الإصلاح يمكنكم الاستخدام الآن", "عاد كل شيء إلى وضعه الطبيعي", "أنهيت العمل الصورة مرفقة"]},
 "other": {"en": ["The lift schedule for visitors is on the notice board", "Who is cleaning the AC training room today", "Meeting about the generator tender tomorrow",
                  "Please do not park near the gate during the event", "Thank you for fixing it earlier by the way happy Eid", "Water bottles are in the store, take what you need",
                  "Fire drill at 11 everyone assemble at the car park", "The light switch ceremony starts at 5", "Pipe band will play at the opening", "Window seats are reserved for guests"],
           "ar": ["جدول المصاعد للزوار على لوحة الإعلانات", "من سينظف قاعة تدريب المكيفات اليوم", "اجتماع بخصوص مناقصة المولد غدًا",
                  "يرجى عدم الوقوف قرب البوابة أثناء الفعالية", "شكرًا على إصلاحه سابقًا وكل عام وأنتم بخير", "زجاجات المياه في المخزن خذوا ما تحتاجون",
                  "تمرين إخلاء الساعة 11 الجميع يتجمع في مواقف السيارات", "حفل افتتاح المبنى يبدأ الساعة الخامسة", "ستعزف الفرقة الموسيقية في الافتتاح", "مقاعد النافذة محجوزة للضيوف"]},
}

OTHER = {
 "en": ["Good morning everyone", "Good evening team", "Lunch is ready in {l}", "Meeting at 3 pm in {l}", "Who has the keys to the van", "Happy birthday to Ahmed",
        "Please send me the attendance sheet", "Traffic is very bad on the highway today", "The match last night was amazing", "Check out my new phone",
        "Shift change at 6 can anyone cover for me", "Visitors arriving at 10 please arrange parking at {l}", "Reminder to submit timesheets by Friday", "Thanks team great work today",
        "Where is the supervisor", "Does anyone know the wifi password", "Delivery truck has arrived", "Tomorrow is a public holiday", "Look at this beautiful sunset",
        "I will be late today", "Can someone bring tea to {l}", "New uniforms will be distributed next week", "Training session starts at 9 tomorrow", "Package for finance is at {l}",
        "Our team won the game", "Please keep the radio channel clear", "Ok copy that", "Cat sitting on the car", "Is the cafeteria open on Saturday", "Security patrol completed no incidents",
        "Please welcome our new colleague Sara", "Anyone going to town after work", "Payroll will be processed on the 25th", "Radio check one two", "Who left the van lights on",
        "Please sign the visitor log at {l}", "Boss is on his way to {l}", "Prayer time in 10 minutes", "Coffee machine order has arrived", "Reminder: safety induction next Monday",
        "Can you resend the schedule", "Weather looks good for the outing", "Parking passes are available at {l}", "Is anyone free to help move boxes", "Congratulations on the promotion"],
 "ar": ["صباح الخير جميعًا", "مساء الخير للفريق", "الغداء جاهز في {l}", "اجتماع الساعة الثالثة في {l}", "من لديه مفاتيح الحافلة", "عيد ميلاد سعيد لأحمد",
        "أرسل لي ورقة الحضور من فضلك", "الزحام شديد على الطريق السريع اليوم", "المباراة البارحة كانت رائعة", "شاهدوا هاتفي الجديد",
        "تبديل الوردية الساعة السادسة هل يستطيع أحد أن يغطي مكاني", "الزوار سيصلون الساعة العاشرة يرجى تجهيز المواقف عند {l}", "تذكير بتسليم كشوف الدوام قبل يوم الجمعة", "شكرًا للفريق عمل رائع اليوم",
        "أين المشرف", "هل يعرف أحد كلمة سر الواي فاي", "وصلت شاحنة التوصيل", "غدًا إجازة رسمية", "انظروا إلى هذا الغروب الجميل",
        "سأتأخر اليوم", "هل يستطيع أحد إحضار الشاي إلى {l}", "سيتم توزيع الزي الجديد الأسبوع القادم", "الدورة التدريبية تبدأ غدًا الساعة التاسعة", "طرد لقسم المالية عند {l}",
        "فريقنا فاز في المباراة", "يرجى إبقاء قناة الراديو خالية", "حسنًا مفهوم", "قطة تجلس على السيارة", "هل الكافتيريا مفتوحة يوم السبت", "انتهت جولة الأمن ولا حوادث",
        "نرحب بزميلتنا الجديدة سارة", "هل أحد ذاهب إلى المدينة بعد الدوام", "سيتم صرف الرواتب يوم 25", "اختبار الراديو واحد اثنان", "من ترك أضواء الحافلة مضاءة",
        "يرجى التوقيع على سجل الزوار عند {l}", "المدير في طريقه إلى {l}", "موعد الصلاة بعد عشر دقائق", "وصل طلب ماكينة القهوة", "تذكير: ورشة السلامة يوم الاثنين القادم",
        "هل يمكنك إعادة إرسال الجدول", "الطقس جميل للرحلة", "بطاقات المواقف متوفرة عند {l}", "هل أحد متفرغ للمساعدة في نقل الصناديق", "مبروك الترقية"],
}
PRE = {"en": ["", "", "", "Urgent: ", "Attention, ", "Guys, ", "Hi, ", "FYI ", "Team, "],
       "ar": ["", "", "", "عاجل: ", "انتباه، ", "يا شباب، ", "مرحبًا، ", "للعلم ", "فريق، "]}
SUF = {"en": ["", "", "", " please check", " pls", " thanks", " (photo attached)", " asap", " over", " copy?"],
       "ar": ["", "", "", " يرجى المعاينة", " من فضلكم", " شكرًا", " (الصورة مرفقة)", " بسرعة", " حول", " مفهوم؟"]}


def style(text, lang):
    t = random.choice(PRE[lang]) + text + random.choice(SUF[lang])
    if lang == "en":
        r = random.random()
        if r < 0.15: t = t.lower()
        elif r < 0.25 and len(t) > 12:  # typo: drop one letter
            i = random.randrange(3, len(t) - 2)
            t = t[:i] + t[i + 1:]
        if random.random() < 0.3: t = t.rstrip(".")
    return t


def main():
    topics = sorted(T)
    random.shuffle(topics)
    dev_topics = set(topics[: max(1, round(len(topics) * 0.2))]) | {"mixed_dev"}
    rows, seen = [], set()

    def add(lang, label, topic, text):
        text = style(text, lang)
        if text in seen: return False
        seen.add(text)
        rows.append({"id": f"t{len(rows)+1:04d}", "lang": lang, "label": label, "topic": topic,
                     "split": "dev" if topic in dev_topics else "train", "text": text})
        return True

    for lang in ("en", "ar"):
        li = 0 if lang == "en" else 2
        for label, k in (("issue", 0), ("done", 1)):
            n = 0
            # mixed hand-written first
            for i, m in enumerate(MIXED[label][lang]):
                topic = "mixed_dev" if i % 5 == 0 else "mixed"
                if add(lang, label, topic, m): n += 1
            while n < PER_CLASS_LANG:
                topic = topics[n % len(topics)]
                tmpl = random.choice(T[topic][li + k])
                if add(lang, label, topic, tmpl.format(l=random.choice(LOC[lang]))): n += 1
        n = 0
        for i, m in enumerate(MIXED["other"][lang]):
            topic = "mixed_dev" if i % 5 == 0 else "mixed"
            if add(lang, "other", topic, m): n += 1
        ot = OTHER[lang]
        while n < PER_CLASS_LANG:
            j = n % len(ot)
            topic = f"chat{j % 9}"
            if topic in ("chat0", "chat5"): topic = "chat_dev"; dev_topics.add(topic)
            if add(lang, "other", topic, ot[j].format(l=random.choice(LOC[lang]))): n += 1
    # fix split for chat_dev topics added late
    for r in rows:
        r["split"] = "dev" if r["topic"] in dev_topics else "train"
    random.shuffle(rows)
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "lang", "label", "topic", "split", "text"])
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(len(rows), "rows ->", OUT)
    print(Counter((r["lang"], r["label"]) for r in rows))
    print(Counter(r["split"] for r in rows), "dev topics:", sorted(dev_topics))


if __name__ == "__main__":
    main()
