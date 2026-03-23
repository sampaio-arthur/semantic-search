export const EXCLUDED_QUERY_TEXTS = new Set([
  "what causes death from Covid-19?",
  "what types of rapid testing for Covid-19 have been developed?",
  "how has COVID-19 affected Canada",
  "what are the guidelines for triaging patients infected with coronavirus?",
  "what are the transmission routes of coronavirus?",
  "are there any clinical trials available for the coronavirus",
  "which biomarkers predict the severe clinical course of 2019-nCOV infection?",
  "what is known about those infected with Covid-19 but are asymptomatic?",
  "Does SARS-CoV-2 have any subtypes, and if so what are they?",
  "How has the COVID-19 pandemic impacted violence in society, including violent crimes?",
  "what are the health outcomes for children who contract COVID-19?",
]);

export function isExcludedQuery(queryText: string): boolean {
  return EXCLUDED_QUERY_TEXTS.has(queryText.trim());
}
