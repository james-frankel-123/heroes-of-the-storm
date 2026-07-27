import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Draft Coach — HotS Fever',
  description:
    'Download the HotS Fever Draft Coach: a native Windows overlay for Heroes of the Storm with live, on-device pick & ban recommendations.',
}

const DOWNLOAD_URL =
  'https://github.com/james-frankel-123/heroes-of-the-storm/releases/download/draft-coach-beta-v0.1/HotsFever-Overlay-beta.zip'
const VERSION = 'Beta v0.1'
const SIZE = '123 MB'

const FEATURES: { title: string; body: string }[] = [
  {
    title: 'Live pick & ban recommendations',
    body: 'The same neural draft engine as the website, running on your machine — instant suggestions for every pick and ban, no network round-trip.',
  },
  {
    title: 'Personalized to your team',
    body: "At the loading screen it reads the match's own files for the real battletags, then weights recommendations by your and your teammates' momentum-adjusted win rates (MAWP).",
  },
  {
    title: 'Auto-filled draft board',
    body: 'Reads the authoritative final draft — heroes, bans, and map — straight from the game, so the board mirrors your real Storm League draft.',
  },
  {
    title: 'Stays out of the way',
    body: 'A translucent, always-on-top overlay you can drag anywhere and collapse. It reads files and (optionally) the screen — it never touches the game process.',
  },
]

const STEPS: string[] = [
  'Download and unzip anywhere (e.g. your Desktop). Nothing to install — the app is self-contained.',
  'In Heroes of the Storm, open Options → Video and set Display Mode to “Borderless Windowed”, so the overlay can appear over the game.',
  'Run HotsFever.Overlay.exe. The overlay appears in the top-left; drag it wherever you like.',
  'Queue for Storm League. During the draft, tap heroes as they’re picked for live recs; at the loading screen the overlay fills in the real draft and your team automatically.',
]

export default function DraftCoachPage() {
  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_#1a1f3a_0%,_#0a0d1f_70%)] text-[#e8d8a0]">
      <div className="max-w-[900px] mx-auto px-6 py-12 space-y-10">

        {/* Hero */}
        <header className="text-center space-y-3">
          <p className="text-xs tracking-[0.4em] text-[#8b9bc8]">HOTS FEVER · FOR WINDOWS</p>
          <h1 className="text-4xl md:text-5xl tracking-[0.25em] text-[#d4b85a] font-light">DRAFT COACH</h1>
          <p className="text-[#8b9bc8] max-w-[620px] mx-auto leading-relaxed">
            A native overlay that gives you the website&apos;s draft engine live, in-game — pick and ban
            recommendations while you actually draft, personalized to your team.
          </p>
        </header>

        {/* Download */}
        <div className="flex flex-col items-center gap-3">
          <a
            href={DOWNLOAD_URL}
            className="inline-flex items-center gap-3 rounded-sm border border-[#d4b85a]/60 bg-[#d4b85a]/10 px-8 py-4 text-lg tracking-[0.15em] text-[#e8d8a0] transition-colors hover:bg-[#d4b85a]/20"
          >
            <span aria-hidden>⬇</span> DOWNLOAD FOR WINDOWS
          </a>
          <p className="text-xs text-[#8b9bc8]">
            {VERSION} · {SIZE} · Windows 10 / 11 (64-bit) · no install required
          </p>
        </div>

        {/* Features */}
        <section className="grid gap-4 sm:grid-cols-2">
          {FEATURES.map((f) => (
            <div key={f.title} className="rounded-sm border border-[#d4b85a]/20 bg-[#0f1430]/60 p-5">
              <h3 className="text-sm tracking-[0.15em] text-[#d4b85a] font-medium mb-2">{f.title}</h3>
              <p className="text-sm text-[#8b9bc8] leading-relaxed">{f.body}</p>
            </div>
          ))}
        </section>

        {/* Setup */}
        <section className="rounded-sm border border-[#d4b85a]/20 bg-[#0f1430]/60 p-6">
          <h2 className="text-sm tracking-[0.2em] text-[#d4b85a] font-light mb-4">GETTING STARTED</h2>
          <ol className="space-y-3">
            {STEPS.map((s, i) => (
              <li key={i} className="flex gap-3 text-sm text-[#cdd6ea] leading-relaxed">
                <span className="flex-none w-6 h-6 rounded-full bg-[#d4b85a]/15 border border-[#d4b85a]/40 text-[#d4b85a] text-xs flex items-center justify-center">
                  {i + 1}
                </span>
                <span>{s}</span>
              </li>
            ))}
          </ol>
        </section>

        {/* Notes */}
        <section className="text-xs text-[#8b9bc8] space-y-2 border-t border-[#d4b85a]/15 pt-6">
          <p>
            <span className="text-[#d4b85a]">Beta.</span> This is an early build — expect rough edges, and
            note the draft screen can shift between game patches. It reads the game&apos;s own replay and
            lobby files and never reads or modifies game memory.
          </p>
          <p>
            <span className="text-[#d4b85a]">Exclusive fullscreen won&apos;t work.</span> No overlay can draw
            over a game in exclusive fullscreen — use Borderless Windowed. The app will warn you if it
            detects exclusive fullscreen.
          </p>
        </section>
      </div>
    </div>
  )
}
