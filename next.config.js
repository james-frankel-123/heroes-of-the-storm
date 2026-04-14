function resolveCommitSha() {
  if (process.env.VERCEL_GIT_COMMIT_SHA) return process.env.VERCEL_GIT_COMMIT_SHA.slice(0, 7)
  if (process.env.NEXT_PUBLIC_COMMIT_SHA) return process.env.NEXT_PUBLIC_COMMIT_SHA
  try {
    return require('child_process').execSync('git rev-parse --short HEAD').toString().trim()
  } catch {
    return 'dev'
  }
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  experimental: {
    optimizePackageImports: ['lucide-react', 'd3'],
  },
  images: {
    domains: [],
  },
  env: {
    NEXT_PUBLIC_COMMIT_SHA: resolveCommitSha(),
  },
}

module.exports = nextConfig
