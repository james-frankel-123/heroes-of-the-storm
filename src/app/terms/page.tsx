import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Terms of Use — HotS Fever',
  description: 'Terms of use for HotS Fever.',
}

export default function TermsPage() {
  return (
    <article className="mx-auto max-w-2xl space-y-4 text-base leading-relaxed">
      <h1 className="text-3xl font-bold">Terms of Use</h1>
      <p>
        Match data and statistics on HotS Fever are provided by{' '}
        <a href="https://www.heroesprofile.com/" className="underline" target="_blank" rel="noopener noreferrer">
          Heroes Profile
        </a>{' '}
        under licence. By using this site you agree to the following:
      </p>
      <ul className="list-disc space-y-2 pl-6">
        <li>
          You may view and use the data here for your own purposes. You may not redistribute it: no
          republishing it as a dataset, dump, feed or API, and no selling or licensing it to anyone.
        </li>
        <li>You may not scrape, crawl or otherwise bulk-extract data from this site.</li>
        <li>You may not use the data to bully, harass or target any player.</li>
        <li>
          Any screenshot, embed or share of data from this site must keep the &ldquo;Data provided by
          Heroes Profile&rdquo; attribution.
        </li>
      </ul>
      <p>
        HotS Fever is a fan-made project. It is not affiliated with, endorsed by, or officially
        connected to Blizzard Entertainment or Heroes of the Storm, or endorsed by Heroes Profile.
        Hero names, map names, artwork and other game assets are the property of Blizzard
        Entertainment.
      </p>
    </article>
  )
}
