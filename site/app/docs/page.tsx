import { product } from "../site-config";
import Link from "next/link";

export const metadata = { title: "Liquid Warp documentation" };

function PurchaseCta() {
  return product.checkoutUrl ? <a className="purchase-cta" href={product.checkoutUrl}>Buy {product.name} — {product.priceLabel}</a> : <Link className="purchase-cta" href="/#gallery">View real demos</Link>;
}

export default function DocsPage() {
  return (
    <main className="docs-page">
      <a className="skip-link" href="#docs-content">Skip to documentation</a>
      <nav className="nav shell" aria-label="Primary navigation"><Link className="wordmark" href="/">Liquid Warp</Link><div className="nav-links"><Link href="/#features">Features</Link><Link href="/#gallery">Demos</Link><Link href="/docs" aria-current="page">Docs</Link></div><PurchaseCta /></nav>
      <article className="docs shell" id="docs-content">
        <p className="eyebrow">Documentation</p><h1>Get Liquid Warp running.</h1><p className="lede">This quick guide covers the supported setup and first live simulation. For the full panel reference, use the included FluidSim Warp user guide.</p>
        <nav className="docs-toc" aria-label="On this page"><strong>On this page</strong><a href="#requirements">Requirements</a><a href="#install">Install</a><a href="#first-sim">First simulation</a><a href="#output">Output and export</a><a href="#troubleshooting">Troubleshooting</a></nav>
        <section id="requirements"><h2>Requirements</h2><ul><li>Blender 5.1</li><li>An NVIDIA GPU with CUDA</li><li><code>warp-lang</code> importable from Blender&apos;s Python</li></ul></section>
        <section id="install"><h2>Install</h2><ol><li>Install and enable the delivered Liquid Warp add-on using Blender&apos;s add-on preferences.</li><li>Open a 3D Viewport, press <kbd>N</kbd>, then choose the <strong>FluidSim</strong> tab.</li><li>Confirm that Blender&apos;s Python can import <code>warp-lang</code> before initializing a solver.</li></ol><p className="callout">Ready state: the FluidSim tab is visible and its solver controls open without a Warp import error.</p></section>
        <section id="first-sim"><h2>First live simulation</h2><ol><li>Save the Blender file, then use <strong>Quick Setup</strong> to create a starting scene.</li><li>In <strong>FluidSim → Simulation</strong>, choose the domain, source, and collider settings.</li><li>Select <strong>Initialize Solver</strong>, then switch <strong>Live: Off</strong> to live mode and play the timeline.</li><li>Particles and the generated surface should update in the viewport. Use the step, reset, and release controls when you need manual playback or a clean restart.</li></ol></section>
        <section id="output"><h2>Surface, cache, and export</h2><p>Use the realtime surfacing controls to generate the liquid mesh and whitewater. Caching is optional: enable it and choose a cache path when you need NPZ frame files. <strong>Export Current Frame</strong> writes the surface as OBJ and particles or whitewater as PLY.</p></section>
        <section id="troubleshooting"><h2>Troubleshooting</h2><ul><li><strong>Warp import error:</strong> verify that <code>warp-lang</code> is importable from Blender&apos;s own Python environment.</li><li><strong>Initialization fails:</strong> confirm Blender 5.1, an available NVIDIA CUDA device, and valid domain/source settings.</li><li><strong>No visible output:</strong> confirm Live is enabled, play the timeline, and check that the source is inside the domain.</li><li><strong>Cache or export fails:</strong> choose a writable destination, then retry the frame.</li><li><strong>Panel missing:</strong> open the 3D Viewport sidebar with <kbd>N</kbd> and select <strong>FluidSim</strong>.</li></ul><p>Reset the solver after structural scene changes; release it before replacing the domain or source.</p></section>
      </article>
    </main>
  );
}
