import katex from "katex";

export default function Formula({ tex, label }: { tex: string; label: string }) {
  const html = katex.renderToString(tex, { displayMode: true, output: "htmlAndMathml", throwOnError: true });
  return <div role="math" aria-label={label} dangerouslySetInnerHTML={{ __html: html }} />;
}
