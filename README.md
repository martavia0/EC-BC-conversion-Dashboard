Dashboard for EC-eBC harmonisation
by: Marta Via (martaviagonzalez@gmail.com)
last update: 2026-08-14

- This dashboard harmonises the EC and eBC measurements into harmonised EC. 
- A harmonised EC measurmeent is arbitrarily defined as an elemental carbon measurement (themo-optical), measured with high-volume samplers, EUSAAR2 thermo-optical protocol, and with a PM2.5 size cut. 
- The dashboard estimates EC-BC conversion based on posterior estimates from the Bayesian model described in Via et al. (in prep.).
- The tool reads the input data, asks the user to introduce the metadata of the measurements and provides an uncertainty estimation for the harmonisation conversion.
- Make sure that your input files contain only two columns: a datetime one, and a measurements one. Check the example in github to make yours similar to that one. 
- Live app: https://ec-bc-conversion-dashboard.onrender.com/
- Hosted on Render's free tier. If the app has been idle, it may take 30-60 seconds to wake up on first load.

Running locally:
--------------
bash
# Clone the repo
git clone https://github.com/martavia0/EC-BC-conversion-Dashboard.git
cd EC-BC-conversion-Dashboard
# Install dependencies
pip install -r requirements.txt
# Run the app
python app.py
--------------
Then open http://127.0.0.1:8050 in your browser.

Citation 
If you use this tool, please cite: Via, M. et al. [Title of the paper]. [Journal], [Year]. [DOI or link]

Maintained by martavia0 (https://github.com/martavia0).
Questions, ideas, or bugs are welcome via GitHub issues.
