import { product } from "./site-config";
import Link from "next/link";

const demos = [
  ["Crown splash", "/crown-splash-preview.png", "Impact-driven particles and liquid surface.", "Blue crown-splash particles and surface inside the simulation domain"],
  ["Dam break", "/dam-break-preview.png", "A source-to-surface release test.", "Dam-break liquid moving through a Blender simulation domain"],
  ["Slosh tank", "/slosh-tank-preview.png", "Contained liquid responding to motion.", "Liquid particles moving inside the slosh-tank demo"],
  ["Tidal lock", "/tidal-lock-preview.png", "Paired sources interacting with moving gates.", "Liquid flowing between moving gates in the tidal-lock demo"],
];

const features = [
  ["Live in the viewport", "Run the sparse-grid PIC/FLIP solver on the GPU without mandatory cache baking."],
  ["Shape the shot", "Control sources, colliders, forces, pressure, viscosity, surfacing, and whitewater in Blender."],
  ["Finish outside Blender", "Optionally cache frames and export liquid surfaces as OBJ plus particles and whitewater as PLY."],
];

function PurchaseCta({ className = "" }: { className?: string }) {
  if (!product.checkoutUrl) {
    return (
      <a className={`purchase-cta ${className}`} href="#gallery">View real demos</a>
    );
  }

  return (
    <a className={`purchase-cta ${className}`} href={product.checkoutUrl}>
      Buy {product.name} — {product.priceLabel}
    </a>
  );
}

export default function Home() {
  return (
    <main>
      <a className="skip-link" href="#content">Skip to product</a>
      <nav className="nav shell" aria-label="Primary navigation">
        <a className="wordmark" href="#top">Liquid Warp</a>
        <div className="nav-links">
          <a href="#features">Features</a>
          <a href="#gallery">Demos</a>
          <Link href="/docs">Docs</Link>
        </div>
        <PurchaseCta className="nav-cta" />
      </nav>

      <section className="hero shell" id="content" aria-labelledby="site-title">
        <div className="hero-intro">
          <p className="eyebrow">GPU liquid simulation for Blender 5.1</p>
          <h1 id="site-title">GPU liquid. Live in Blender.</h1>
        </div>
        <div className="hero-visual">
          <img src="/liquid-warp-preview.png" alt="Liquid Warp crown splash simulation in Blender" />
          <div className="visual-caption"><span>ACTUAL ADD-ON OUTPUT</span><strong>Crown splash · live viewport</strong></div>
        </div>
        <div className="hero-details">
          <p className="lede">
            Liquid Warp is the FluidSim Warp add-on for artists who want cinematic liquid,
            live viewport playback, and a GPU-resident workflow powered by NVIDIA Warp.
          </p>
          <div className="hero-actions">
            <PurchaseCta />
            <Link className="text-link" href="/docs">Read the docs <span aria-hidden="true">→</span></Link>
          </div>
          <p className="launch-note"><strong>Launch preview</strong><span>{product.priceLabel} · checkout not yet configured</span></p>
        </div>
      </section>

      <section className="positioning shell" aria-label="Product fit">
        <p>For Blender artists building splashes, tanks, pours, and moving-water shots.</p>
        <span>Live iteration first. Cache only when the shot needs it.</span>
      </section>

      <section className="proof-band" aria-label="Product highlights">
        <div className="shell proof-grid">
          <p><strong>Blender 5.1</strong><span>Target runtime</span></p>
          <p><strong>NVIDIA CUDA</strong><span>GPU requirement</span></p>
          <p><strong>Warp-powered</strong><span>GPU simulation</span></p>
          <p><strong>OBJ + PLY</strong><span>Export options</span></p>
        </div>
      </section>

      <section className="section shell" id="features" aria-labelledby="features-title">
        <div className="section-heading"><p className="eyebrow">Built for the shot</p><h2 id="features-title">Get from source to surface without leaving the flow.</h2></div>
        <div className="feature-grid">
          {features.map(([title, description], index) => <article key={title} className="feature-card"><span>0{index + 1}</span><h3>{title}</h3><p>{description}</p></article>)}
        </div>
      </section>

      <section className="section shell gallery-section" id="gallery" aria-labelledby="gallery-title">
        <div className="section-heading"><p className="eyebrow">Real demo scenes</p><h2 id="gallery-title">Built to test liquid in motion.</h2></div>
        <div className="demo-grid">
          {demos.map(([title, image, description, alt]) => <figure key={title}><div className="demo-media"><img src={image} alt={alt} /><span>Actual add-on output</span></div><figcaption><strong>{title}</strong><span>{description}</span></figcaption></figure>)}
        </div>
      </section>

      <section className="section shell workflow" aria-labelledby="workflow-title">
        <div><p className="eyebrow">A direct workflow</p><h2 id="workflow-title">Set the domain. Start live. Shape the result.</h2></div>
        <ol>
          <li><strong>1. Quick Setup</strong><span>Create the scene controls in the FluidSim tab.</span></li>
          <li><strong>2. Drive the simulation</strong><span>Use sources, colliders, forces, and solver controls while playing in the viewport.</span></li>
          <li><strong>3. Surface and deliver</strong><span>Generate the liquid surface, add whitewater, then cache or export only when needed.</span></li>
        </ol>
      </section>

      <section className="section shell requirements" aria-labelledby="requirements-title">
        <div><p className="eyebrow">Before you buy</p><h2 id="requirements-title">System requirements</h2></div>
        <ul><li>Blender 5.1</li><li>NVIDIA GPU with CUDA</li><li><code>warp-lang</code> importable from Blender&apos;s Python</li></ul>
      </section>

      <section className="section shell requirements" aria-labelledby="included-title">
        <div><p className="eyebrow">What you get</p><h2 id="included-title">One workflow from simulation to delivery.</h2></div>
        <ul><li>FluidSim Warp add-on for Blender 5.1</li><li>Realtime surfacing and whitewater controls</li><li>Optional NPZ frame cache plus OBJ and PLY export</li><li>Installation and first-simulation documentation</li></ul>
      </section>

      <section className="section shell faq" aria-labelledby="faq-title">
        <div className="section-heading"><p className="eyebrow">Straight answers</p><h2 id="faq-title">Frequently asked questions</h2></div>
        <details><summary>Does Liquid Warp require baking?</summary><p>No. The live solver is designed for interactive viewport playback; optional caching is available when you need it.</p></details>
        <details><summary>What does it export?</summary><p>Export Current Frame writes the liquid surface as OBJ and particles or whitewater as PLY.</p></details>
        <details><summary>What does the add-on include?</summary><p>Live GPU simulation, surfacing, whitewater, optional NPZ frame caching, and OBJ or PLY export. Final license and update terms will appear at checkout before launch.</p></details>
      </section>

      <section className="closing shell" aria-labelledby="closing-title"><p className="eyebrow">Make the water yours</p><h2 id="closing-title">Stay live until the shot is right.</h2><p>{product.priceLabel} · final license terms available at launch</p><PurchaseCta /></section>
      <footer className="footer shell"><span>Liquid Warp · FluidSim Warp for Blender 5.1</span><Link href="/docs">Documentation</Link></footer>
    </main>
  );
}
