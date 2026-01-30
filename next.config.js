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
}

module.exports = nextConfig
