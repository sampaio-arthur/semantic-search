"""Fixed ideal answers for the 39 valid BEIR trec-covid queries.

These are part of the experimental configuration and must not be edited
by users at runtime — reproducibility requires deterministic ground truth.

Keys are the original BEIR query_ids (not sequential 1-39).
"""

from __future__ import annotations

IDEAL_ANSWERS: dict[str, str] = {
    "1": "SARS-CoV-2 likely originated in bats, with possible intermediate host, linked to Wuhan seafood market in late 2019.",
    "2": "Higher temperatures and humidity may reduce transmission, but evidence is inconclusive; the virus persists in varied climates.",
    "3": "Most infected develop antibodies, but duration of immunity is uncertain. Limited cross-protection from other coronaviruses is possible but not confirmed.",
    "5": "Remdesivir, chloroquine, lopinavir/ritonavir, and interferon showed activity in animal models against SARS-CoV or SARS-CoV-2.",
    "7": "Yes, ELISA and rapid lateral flow assays detect IgM/IgG antibodies to SARS-CoV-2, though sensitivity and specificity vary.",
    "8": "Limited testing capacity caused significant underreporting; seroprevalence studies suggest actual cases far exceed confirmed counts.",
    "10": "Yes, social distancing measures significantly reduced transmission rates and flattened infection curves in multiple countries.",
    "12": "Hospital: PPE, negative pressure rooms, strict isolation. Home: separate room, dedicated bathroom, mask use, hand hygiene, surface disinfection.",
    "14": "Super-spreading events linked to enclosed spaces, poor ventilation, prolonged close contact; a minority of infected cause majority of transmissions.",
    "15": "SARS-CoV-2 survives hours on copper, up to 24h on cardboard, and 2-3 days on plastic and stainless steel surfaces.",
    "16": "Viable virus detected up to 72h on plastic/steel, 24h on cardboard, 4h on copper; UV light and disinfectants reduce viability.",
    "18": "N95 respirators offer highest protection; surgical masks reduce transmission; cloth masks provide moderate protection depending on material and fit.",
    "19": "Alcohol-based hand sanitizers with at least 60% ethanol or 70% isopropanol effectively inactivate SARS-CoV-2.",
    "20": "Evidence is mixed; ACE2 is the viral receptor, but ACE inhibitors may not increase risk. Major societies recommend continuing treatment.",
    "21": "Overall IFR estimated at 0.5-1%; mortality significantly higher in elderly (>70) and those with comorbidities.",
    "22": "Yes, myocarditis, arrhythmias, acute cardiac injury, and heart failure reported, especially in severe cases.",
    "23": "Hypertension is a major risk factor for severe COVID-19; associated with higher ICU admission, ventilation, and mortality.",
    "24": "Diabetes increases risk of severe illness, ICU admission, ARDS, and death due to impaired immune response and chronic inflammation.",
    "26": "Fever, dry cough, and fatigue are most common; loss of taste/smell, sore throat, and myalgia also reported.",
    "28": "Early in-vitro activity was promising, but randomized trials showed no significant benefit; potential cardiac side effects noted.",
    "29": "Spike-ACE2 interaction is the primary target; viral proteases (3CLpro, PLpro) and RdRp are druggable; some overlap with approved drug targets.",
    "30": "Remdesivir showed modest reduction in recovery time in hospitalized patients; limited efficacy on mortality; FDA granted EUA.",
    "31": "COVID-19 has higher mortality, longer incubation, higher rate of severe pneumonia; flu has established vaccines and treatments.",
    "33": "mRNA (Moderna, Pfizer), adenoviral vector (AstraZeneca, J&J), inactivated (Sinovac), and protein subunit vaccines were tested.",
    "34": "Long COVID includes fatigue, dyspnea, cognitive impairment, chest pain, and organ damage (pulmonary fibrosis, cardiac, neurological).",
    "35": "CORD-19, Johns Hopkins CSSE, GISAID genome sequences, and WHO situation reports are key public datasets.",
    "36": "Homotrimeric glycoprotein with S1 (RBD binds ACE2) and S2 (fusion) subunits; solved by cryo-EM at ~3.5\u00c5 resolution.",
    "37": "Closely related to bat coronavirus RaTG13 (~96% similarity); forms distinct clade within betacoronaviruses.",
    "38": "Viral entry via ACE2 triggers innate immune response; dysregulated inflammation leads to ARDS, coagulopathy, and multi-organ failure.",
    "39": "Excessive release of IL-6, TNF-\u03b1, IL-1\u03b2 causes hyperinflammation, vascular permeability, organ damage; major driver of severe mortality.",
    "40": "D614G became dominant early; spike mutations drive variants of concern; ~2 substitutions/month; most mutations neutral or deleterious.",
    "41": "Disproportionately higher infection/death rates due to socioeconomic disparities, comorbidity prevalence, and healthcare access gaps.",
    "42": "Low vitamin D correlates with higher severity; supplementation may support immunity, but clinical evidence for prevention/treatment is insufficient.",
    "44": "Masks reduce transmission by 50-80% depending on type and compliance; universal masking significantly lowers community spread.",
    "45": "Increased rates of anxiety, depression, PTSD, and substance abuse; healthcare workers and isolated populations particularly affected.",
    "46": "RECOVERY trial: dexamethasone reduced mortality by ~1/3 in ventilated patients, ~1/5 on oxygen; no benefit in mild cases.",
    "48": "Benefits: educational continuity, mental health. Risks: transmission clusters; mitigation depends on local prevalence and safety measures.",
    "49": "Most develop neutralizing antibodies and T-cell responses; protection wanes over time; reinfection documented with new variants.",
    "50": "mRNA vaccines (Pfizer, Moderna) encode spike protein; ~95% efficacy in phase 3; first mRNA vaccines authorized for human use.",
}
