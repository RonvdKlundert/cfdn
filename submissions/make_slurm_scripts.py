import yaml
import os
import json
import re

# Point this to the template file I sent. This contains everything to be populated in between three sets of dashes (e.g. ---n---)
template='/home/klundert/utils/template_2folds.sh'
# Point this to the yaml I sent. This contains all the things that are going to be the same for each job (i.e. the email, the requested resources etc).
myyaml='/home/klundert/utils/examp_yaml.yml'
#point this to where you want the jobscripts output
out_dir='/home/klundert/JOBS/'


# Define a class for populating template sh file given a yaml file
class Script_Populator:
    
    """Script_Populator
    Class for populating a script.
    
    The idea is to take a template file and populate it with information contained in a yaml file.
    
    
    """
    
    
    def __init__(self,yaml_file,template_file,out_dir,jobname='myfmriprep',suffix='.sh',**kwargs):
        
        """
        Initialise the class.
        
        Parameters
        ----------
        yaml_file: yaml file containing the information to be fed into the jobscript.
        template_file: template file for the jobscript 
        out_dir: Where to save the populated script.
        jobname: The name given to the job.
        
        
        An additional 'supdict' dictionary can be provided in kwargs to populate additional information.
        This is useful in the case where the script needs to be populated on the fly.
        
        Parameters
        ----------
        
        self.outfile: Script output location.
        self.working_string: The unpopulated template script.
        
        """
        self.jobname=jobname # Name to be given to job.
        
        self.yaml_file=yaml_file
        
        
        with open(yaml_file, 'r') as f:
            self.yaml = yaml.safe_load(f)
        
        # Append the supdict if it exists.
        if 'supdict' in kwargs:
            supdict = kwargs.get('supdict')
            self.yaml={**self.yaml, **supdict}
        
        # Read the jobscript template into memory.
        self.jobscript = open(template_file)
        self.working_string = self.jobscript.read()
        self.jobscript.close()

        subject = supdict['---subject---']
        slice_n = supdict['---data_portion---']
        self.outfile=os.path.join(out_dir, f'sub-{subject}_slice-{slice_n}' + suffix)
        
        
    def populate(self):
        
        """ populate
        
        Populates the jobscript with items from the yaml
        
        Returns
        ----------
        self.working_string: populated script.
        
        """
        
        
        for e in self.yaml:
            rS = re.compile(e)
            self.working_string = re.sub(rS, self.yaml[e], self.working_string)
            
    def writeout(self):
        
        """  writeout
       
        Writes out the jobscript file to the outfile location.
        
        """
        
        
        of = open(self.outfile, 'w')
        of.write(self.working_string)
        of.close()
        
    def execute(self,execute_type='sbatch'):
        
        """ execute
        Executes the script.
        """
        
        os.system(execute_type + " " + self.outfile)
        print('{job} sent to SLURM'.format(job=self.jobname))
        
# Create a list of dictionaries of arguments that need to be populated dynamically.
# Here we create a dictionary for every instance of subject and data portion

#new_subs = [6,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,
#        89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101,
#       102, 103, 104, 105, 157, 164, 173]

#prf_subs = [2, 7, 8, 13, 22, 25, 26, 28, 31]

mysuppdicts=[dict({'---subject---':str(p),'---data_portion---':str(dp)}) for p in range(2) for dp in range(357)]

# range(2) range(200)
# 174 and 8

# Now we run through a big loop that populates the template script based on everything in each dictionary and writes it to a file.
#You can uncomment the last line to send them all to slurm (check first though).
for cdict in mysuppdicts:
    x=Script_Populator(myyaml,template,out_dir,jobname=json.dumps(cdict),supdict=cdict)
    x.populate()
    x.writeout()
    # Only uncomment the next line after checking the files have populated correctly
    x.execute()

