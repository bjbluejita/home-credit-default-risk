'''
@Project: home-credit-default-risk
@Package 
@author: ly
@date Date: 2019年04月25日 11:55
@Description: 
@URL: https://www.jianshu.com/p/488750ca69f0
@version: V1.0
'''
import click

@click.command()
@click.option( '--count', default=1, help='Number of greetings' )
@click.option( '--name', prompt='Your name', help='The person to geet' )
def hello( count, name ):
    for x in range( count ):
        click.echo( "Hello %s!" % name )

if __name__ == '__main__':
    hello()