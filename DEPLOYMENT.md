# Deployment Guide

## Prerequisites

- GitHub account
- Vercel account (sign up at [vercel.com](https://vercel.com))
- Node.js 18+ installed locally

## Local Development

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

## Deploy to Vercel

### Option 1: Deploy via Vercel Dashboard (Recommended)

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub
2. Click "Add New Project"
3. Import your GitHub repository: `james-frankel-123/heroes-of-the-storm`
4. Configure project settings:
   - **Framework Preset**: Next.js
   - **Root Directory**: `./` (leave default)
   - **Build Command**: `npm run build` (default)
   - **Output Directory**: `.next` (default)
5. Click "Deploy"

### Option 2: Deploy via Vercel CLI

1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```

2. Login to Vercel:
   ```bash
   vercel login
   ```

3. Deploy:
   ```bash
   vercel
   ```

4. Follow the prompts:
   - Set up and deploy? **Y**
   - Which scope? Select your account
   - Link to existing project? **N**
   - What's your project's name? **heroes-of-the-storm**
   - In which directory is your code located? **.**
   - Want to override the settings? **N**

5. For production deployment:
   ```bash
   vercel --prod
   ```

## Environment Variables (Optional)

If you need to add environment variables:

1. In Vercel Dashboard:
   - Go to your project settings
   - Navigate to "Environment Variables"
   - Add variables as needed

2. Via CLI:
   ```bash
   vercel env add <NAME>
   ```

## Data Updates

The app currently uses a static JSON file at `data/player-stats.json`. To update player statistics:

1. Replace `data/player-stats.json` with updated data
2. Commit and push:
   ```bash
   git add data/player-stats.json
   git commit -m "Update player statistics"
   git push
   ```
3. Vercel will automatically redeploy

## Future Backend Options

For a fully dynamic backend, consider:

1. **Vercel Postgres** - Built-in PostgreSQL database
2. **Supabase** - Open-source Firebase alternative
3. **MongoDB Atlas** - NoSQL database
4. **API Routes** - Extend Next.js API routes to fetch from Heroes Profile API

## Monitoring

- View deployment logs in Vercel Dashboard
- Check analytics at: `https://vercel.com/[your-username]/heroes-of-the-storm/analytics`
- Monitor performance at: `https://vercel.com/[your-username]/heroes-of-the-storm/speed-insights`

## Custom Domain (Optional)

1. In Vercel Dashboard, go to project settings
2. Navigate to "Domains"
3. Add your custom domain
4. Follow DNS configuration instructions

## Troubleshooting

### Build fails on Vercel

- Check build logs in Vercel Dashboard
- Ensure all dependencies are in `package.json`
- Verify `next.config.js` is properly configured

### API routes return 404

- Check that `src/app/api/data/route.ts` exists
- Verify data file is at `data/player-stats.json`
- Check Vercel function logs

### Data not loading

- Verify `data/player-stats.json` is committed to git
- Check browser console for API errors
- Verify API route is accessible at `/api/data`

## Support

For issues, visit: https://github.com/james-frankel-123/heroes-of-the-storm/issues
