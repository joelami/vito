"""
Short, factual preseason context per FBS team for the 2026 season --
NOT a ranking, NOT a model input, NOT the source article's own analysis.
Deliberately extracted-facts-only (coaching changes, transfers, records,
injuries) rather than the journalist's evaluative commentary, per the app
owner's explicit instruction (2026-08-25) after a real copyright-reproduction
concern was raised about copying a paywalled journalist's original written
analysis into this codebase for redistribution. Facts aren't copyrightable;
his specific phrasing/opinions are -- so every note here is independently
reworded, never copied, and never includes the article's numeric ranking
order (this app deliberately doesn't use poll rankings as a signal or
display element at all -- see docs/METHODOLOGY.md's NCAAF section).

Source: 2026 preseason CFB rankings, The Athletic (Chris Vannini),
published 2026-08-24 -- used here only as a pointer to real, independently-
verifiable facts (coaching hires, transfer portal moves, records), not
reproduced as content. This is a one-time preseason snapshot, not
continuously updated -- will go stale as the season progresses.

Keyed by the EXACT franchise strings this dataset uses (sports.cfb.config's
FBS_TEAMS / canonical_team format) so lookups match directly -- verify a
key against that list before adding it.

Note on ambiguous keys: a handful of schools appear in the dataset's FBS
list under more than one historical name-spelling variant. Where that
happened, the variant judged most likely current/canonical was used here;
see the extraction report for the alternates that were NOT used:
  - TCU: used "TCU Horned Frogs" (not "Texas Christian Horned Frogs")
  - San Jose State: used "San José State Spartans" (not "San Jose State
    Spartans" or "San José St Spartans")
  - Florida International: used "Florida International Panthers" (not
    "Florida Intl Golden Panthers")
  - Southern Miss: used "Southern Mississippi Golden Eagles" (not
    "Southern Miss Golden Eagles")
  - Appalachian State: used "Appalachian State Mountaineers" (not
    "App State Mountaineers")
"""

TEAM_STORYLINES = {
    "Indiana Hoosiers": "Defending national champion. Added RB Turbo Richard and WR Nick Marsh; new starting QB Josh Hoover.",
    "Ohio State Buckeyes": "Lost a significant number of players on defense from last season's roster.",
    "Georgia Bulldogs": "Offense lacked explosive plays during the 2025 season.",
    "Oregon Ducks": "Six of head coach Dan Lanning's eight career losses have come against eventual national title game participants.",
    "Notre Dame Fighting Irish": "Returns the most production of any FBS team this season.",
    "Texas Longhorns": "QB Arch Manning and WR Cam Coleman return.",
    "Miami Hurricanes": "Reached the CFP championship game as runner-up last season.",
    "LSU Tigers": "New head coach Lane Kiffin, in his first season leading the program.",
    "Texas Tech Red Raiders": "Returning QB Will Hammond.",
    "Oklahoma Sooners": "Returning QB John Mateer.",
    "BYU Cougars": "Second-year QB Bear Bachmeier returns. Missed the CFP field narrowly in each of the last two seasons.",
    "Ole Miss Rebels": "QB Trinidad Chambliss and RB Kewan Lacy return.",
    "Texas A&M Aggies": "QB Marcel Reed returns. Rebuilt offensive and defensive lines.",
    "USC Trojans": "QB Jayden Maiava returns with a new receiving corps and new defensive coordinator Gary Patterson.",
    "Alabama Crimson Tide": "New starting QB. Had rushing offense issues last season.",
    "Michigan Wolverines": "Won nine games last season amid coaching turmoil. New head coach Kyle Whittingham. Second-year QB Bryce Underwood.",
    "SMU Mustangs": "QB Kevin Jennings returns with a new group of receivers.",
    "Louisville Cardinals": "RB Isaac Brown returns. Transfer QB Lincoln Kienholz joins the team.",
    "Houston Cougars": "New coach Willie Fritz. QB Conner Weigman returns. RB Makhi Hughes transferred in after playing for Fritz at Tulane and later at Oregon.",
    "Tennessee Volunteers": "New starting QB. Defense had multiple season-ending injuries in August.",
    "Boise State Broncos": "Joined the relaunched Pac-12 conference.",
    "Oklahoma State Cowboys": "Has not won a Big 12 game since 2023. Added staff and players from North Texas, which had the nation's No. 1 offense last season.",
    "Washington Huskies": "QB Demond Williams Jr. returns.",
    "Iowa Hawkeyes": "Turnover at QB and on defense. One of five teams (with Ohio State, Alabama, Michigan, Georgia) to win at least eight games in every non-pandemic-altered season since 2015.",
    "Navy Midshipmen": "QB Braxton Woodson returns.",
    "Utah Utes": "New head coach Morgan Scalley. QB Devon Dampier returns. Entirely new starting offensive and defensive lines.",
    "TCU Horned Frogs": "Harvard transfer QB Jaden Craig takes over as starter.",
    "Arizona Wildcats": "QB Noah Fifita returns after a nine-win season.",
    "Penn State Nittany Lions": "New head coach Matt Campbell brought players over from Iowa State, including QB Rocco Becht.",
    "Illinois Fighting Illini": "First back-to-back nine-win seasons in school history. QB change to East Carolina transfer Katin Houser under coach Bret Bielema.",
    "Florida Gators": "New head coach Jon Sumrall, who won three conference titles in four seasons as a head coach. QB Aaron Philo and OC Buster Faulkner both came over from Georgia Tech.",
    "South Carolina Gamecocks": "Coach Shane Beamer rebuilt the offensive line. QB LaNorris Sellers and DE Dylan Stewart return.",
    "Clemson Tigers": "Went 7-6 last season with nine players drafted to the NFL.",
    "Virginia Cavaliers": "Reached the ACC title game last season. Coach Tony Elliott. QB Beau Pribula and RB Peyton Lewis return.",
    "Missouri Tigers": "QB change to Austin Simmons. RB Ahmad Hardy recovering after being shot in the leg.",
    "Auburn Tigers": "New head coach Alex Golesh brought QB Byrum Brown and other players over from South Florida.",
    "California Golden Bears": "QB Jaron-Keawe Sagapolutele returns.",
    "Pittsburgh Panthers": "QB Mason Heintschel returns after his first season as starter.",
    "Minnesota Golden Gophers": "QB Drake Lindsey returns.",
    "Georgia Tech Yellow Jackets": "QB Haynes King departed. Michigan transfer Justice Haynes now leads the run game.",
    "New Mexico Lobos": "Won nine games in Jason Eck's first season as coach, tied atop the Mountain West. LB Jaxton Eck returns.",
    "James Madison Dukes": "New head coach Billy Napier inherited a team coming off a CFP appearance. RB George Pettaway returns.",
    "San Diego State Aztecs": "Won nine games last season behind its defense. Coach Sean Lewis. Moving to the relaunched Pac-12.",
    "UNLV Rebels": "Coach Dan Mullen led the team to last season's Mountain West Championship Game. QB Jackson Arnold joins the team.",
    "Arizona State Sun Devils": "New starting QB Cutter Boley, with several new transfers on the roster.",
    "Kansas State Wildcats": "New head coach Collin Klein, a Kansas State alum. QB Avery Johnson returns.",
    "Virginia Tech Hokies": "New head coach James Franklin brought over Penn State contributors, including QB Ethan Grunkemeyer.",
    "NC State Wolfpack": "QB CJ Bailey returns and brought in former high school teammates as transfers. RB Duke Scott.",
    "Florida State Seminoles": "Coach Mike Norvell. QB Ashton Daniels. WR Duce Robinson returns.",
    "Wake Forest Demon Deacons": "Won nine games last season. New starting QB Gio Lopez.",
    "Vanderbilt Commodores": "QB Diego Pavia departed. Five-star freshman QB Jared Curtis takes over as starter.",
    "UTSA Roadrunners": "QB Owen McCown returns.",
    "Tulane Green Wave": "New head coach Will Hall, following a CFP appearance last season.",
    "Wisconsin Badgers": "Coach Luke Fickell. Old Dominion transfer QB Colton Joseph joins the team.",
    "Baylor Bears": "QB change to DJ Lagway. Coach Dave Aranda.",
    "Western Michigan Broncos": "Last season's MAC champions, winning 10 of their last 11 games. QB Broc Lowry and 1,000-yard rusher Jalen Buckley return.",
    "Fresno State Bulldogs": "Coach Matt Entz, who won nine games last season.",
    "Nebraska Cornhuskers": "Third-most returning production in the country. New starters at QB (Anthony Colandrea) and RB, plus a new defensive coordinator.",
    "Duke Blue Devils": "Defending ACC champions. RB Nate Sheppard returns.",
    "Northwestern Wildcats": "Coach David Braun. New offensive coordinator Chip Kelly. QB Aidan Chiles.",
    "South Florida Bulls": "New head coach Brian Hartline brought in the No. 1 Group of Five transfer portal class, including QB Michael Van Buren.",
    "Memphis Tigers": "New head coach Charles Huff, previously at Marshall and Southern Miss. QB battle involving Air Noland.",
    "West Virginia Mountaineers": "Oklahoma transfer QB Michael Hawkins Jr. and Jacksonville State transfer RB Cam Cook join the team.",
    "Kentucky Wildcats": "New head coach Will Stein. QB Kenny Minchey and RB CJ Baxter.",
    "UCLA Bruins": "New head coach Bob Chesney kept QB Nico Iamaleava and brought over players from James Madison, including RB Wayne Knight.",
    "Army Black Knights": "QB Cale Hellums returns. Defense replaces eight starters.",
    "Hawai'i Rainbow Warriors": "Coming off a nine-win season. QB Micah Alejado.",
    "Mississippi State Bulldogs": "QB Kamario Taylor. Rest of the offense is mostly new.",
    "UCF Knights": "James Madison transfer QB Alonza Barnett III joins the team.",
    "Cincinnati Bearcats": "Georgia Southern transfer QB JC French IV joins the team.",
    "Michigan State Spartans": "New head coach Pat Fitzgerald, returning to a college sideline.",
    "Maryland Terrapins": "QB Malik Washington enters year two as starter. Star edge rusher Zahir Mathis is out for the season.",
    "Kansas Jayhawks": "RB Dylan Edwards returns. Offense is mostly new, with no settled starting QB.",
    "North Carolina Tar Heels": "New offensive coordinator Bobby Petrino under head coach Bill Belichick.",
    "Arkansas Razorbacks": "New head coach Ryan Silverfield.",
    "Colorado Buffaloes": "Coach Deion Sanders. New offensive coordinator Brennan Marion.",
    "Syracuse Orange": "QB Steve Angeli is healthy. New running backs and receivers.",
    "Rutgers Scarlet Knights": "RB Antwan Raymond and WR KJ Duff return. Defense replaces all of its starters.",
    "East Carolina Pirates": "Lost significant talent to the transfer portal. Transfer QB Emory Williams (from Miami) and OC Jordan Davis (from North Texas) join the staff.",
    "Troy Trojans": "Last season's Sun Belt West champions. QB Goose Crowder returns.",
    "Texas State Bobcats": "Coming off three consecutive bowl game appearances. QB Brad Jackson.",
    "Washington State Cougars": "Third head coach in three years. UC Davis transfer QB Caden Pinnick, an FCS freshman All-American, joins the team.",
    "Miami (OH) RedHawks": "Three consecutive MAC Championship Game appearances. Brought in the MAC's No. 1 transfer portal class, including 1,800-yard rusher Rodney Nelson.",
    "Old Dominion Monarchs": "RB Devin Roche.",
    "Liberty Flames": "Added RBs Kanye Udoh (Arizona State transfer) and Kam Davis (Florida State transfer). Coach Jamey Chadwell is recovering from January surgery.",
    "Temple Owls": "Coach KC Keeler nearly reached a bowl game in his first season. Penn State transfer QB Jaxon Smolik joins the team.",
    "North Texas Mean Green": "New head coach Neal Brown. Former West Virginia RB Jaheim White transfers in.",
    "Western Kentucky Hilltoppers": "Coach Tyson Helton. QB Rodney Tisdale Jr. returns.",
    "Jacksonville State Gamecocks": "Reached the CUSA title game in coach Charles Kelly's first season. Veteran starting QB Caden Creel returns.",
    "Central Michigan Chippewas": "New head coach Matt Drinkall, previously at Army.",
    "Toledo Rockets": "New head coach Mike Jacobs. DE Andrew Zock won FCS defensive player of the year at Mercer.",
    "Stanford Cardinal": "New head coach Tavita Pritchard. Former Michigan QB Davis Warren expected to start. RB Micah Ford.",
    "Boston College Eagles": "Division II Saginaw Valley State transfer QB Mason McKenzie joins the team. TE Kaelan Chudzinski is out for the year.",
    "Ohio Bobcats": "New head coach John Hauser, the third straight Ohio coach promoted into the job internally.",
    "Purdue Boilermakers": "QB Ryan Browne and LB Charles Correa return. Went 2-10 last season.",
    "Iowa State Cyclones": "Only 13 total starts return; much of the roster followed former coach Matt Campbell to Penn State. New head coach Jimmy Rogers brought over 12 players from Washington State's defense.",
    "Florida International Panthers": "Coach Willie Simmons won seven games in his first season. Transfer QB JJ Kohl (6-foot-7) joins from Appalachian State.",
    "Marshall Thundering Herd": "QB Carlos Del Rio-Wilson returns.",
    "UConn Huskies": "New head coach Jason Candle, previously at Toledo. Former Tennessee QB Jake Merklinger joins the team.",
    "Georgia Southern Eagles": "QB Max Johnson, in his seventh season of college football, returns. Lost all 11 offensive starters.",
    "Utah State Aggies": "Went 6-7 last season. RB Javen Jacobs returns.",
    "Air Force Falcons": "Coming off consecutive losing seasons. QB Liam Szarka.",
    "Arkansas State Red Wolves": "Coach Butch Jones has led the team to three consecutive bowl games.",
    "Louisiana Tech Bulldogs": "Won eight games last season. QB Blake Baker returns from an ACL injury. Moving to the Sun Belt Conference.",
    "Eastern Michigan Eagles": "Most returning production in the MAC. QB Noah Kim returns.",
    "Buffalo Bulls": "Offense is working out its starting lineup this preseason.",
    "Florida Atlantic Owls": "QB Caden Veltkamp threw for more than 3,600 yards last season. Team had a turnover margin of minus-21 last season, the worst nationally by 8 turnovers.",
    "Tulsa Golden Hurricane": "QB Baylor Hayes. CB Elijah Green.",
    "Appalachian State Mountaineers": "Heavy roster turnover for a second straight offseason. Former Purdue and Arkansas QB Malachi Singleton transfers in.",
    "Louisiana Ragin' Cajuns": "QB Lunch Winfield returns. Closed last regular season with four consecutive wins.",
    "Rice Owls": "Former UCF QB Jacurri Brown transfers in. RB Quinton Jackson.",
    "Colorado State Rams": "New head coach Jim Mora. Former Oklahoma State QB Hauss Hejny transfers in.",
    "Oregon State Beavers": "New head coach JaMarcus Shephard. QB Maalik Murphy. Went 2-10 last season.",
    "New Mexico State Aggies": "RB James Jones (Delaware State transfer) and TE Josiah Thomas (Western Carolina transfer) join the team. Was among the worst rushing teams in the country last season.",
    "Coastal Carolina Chanticleers": "New head coach Ryan Beard, with more than 50 newcomers on the roster. LBs Se'Von McDowell and Tray Brown return.",
    "South Alabama Jaguars": "QB Bishop Davenport returns. New defensive coordinator Todd Orlando.",
    "Ball State Cardinals": "Coach Mike Uremovich. Offensive line allowed 49 sacks over 12 games last season.",
    "Bowling Green Falcons": "Former Oregon QB Austin Novosad transfers in. RB Austyn Dendy.",
    "Akron Zips": "RB Jordan Gant. QB Reese Poffenbarger.",
    "Kent State Golden Flashes": "Went 5-7 last season. Interim coach Mark Carney was given the full-time job. QB Dru DeShields returns.",
    "Nevada Wolf Pack": "Consecutive three-win seasons under coach Jeff Choate.",
    "Southern Mississippi Golden Eagles": "New head coach Blake Anderson, previously at Arkansas State and Utah State.",
    "San José State Spartans": "Fell to 3-9 last season. Hawaii transfer QB Luke Weaver takes over as starter.",
    "Wyoming Cowboys": "New offensive coordinator Christian Taylor, previously at William & Mary.",
    "UAB Blazers": "QB Ryder Burton. Coach Alex Mortensen was promoted to the job after last season.",
    "Middle Tennessee Blue Raiders": "Coach Derek Mason enters year three after consecutive 3-9 seasons. QB Roman Gagliano returns.",
    "Northern Illinois Huskies": "Moving to the Mountain West Conference. Interim coach Rob Harley. No defensive starters return.",
    "Louisiana Monroe Warhawks": "Roster is largely new again this offseason.",
    "UTEP Miners": "Incarnate Word transfer QB EJ Colson joins the team.",
    "Sam Houston Bearkats": "Dropped to 2-10 last season. Returning to its home stadium after more than a year of renovation.",
    "Georgia State Panthers": "Added 34 transfers after a 1-11 season. QB Cameran Brown returns.",
    "Massachusetts Minutemen": "Went 0-12 last season, finishing 134th nationally in both scoring offense and scoring defense. QB William Watson III transfers in from Virginia Tech. Two FCS teams on this year's schedule.",
}
