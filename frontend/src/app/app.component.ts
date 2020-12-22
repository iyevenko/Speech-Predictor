import {Component, ElementRef, HostListener, ViewChild} from '@angular/core';
import {TextService} from './text.service';
import {MatInputModule} from '@angular/material/input';
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'client';
  playerName;
  responseText;
  public searchStr = '';

  constructor(private textService: TextService) {
  this.playerName = '';
  this.responseText = '';
  }

  @ViewChild('textInput') textInput: ElementRef | undefined;

  @HostListener('document:click', ['$event'])

  documentClick(event: MouseEvent) {
    console.log(this.searchStr);
    if (!(this.textInput == undefined)){
      this.textInput.nativeElement.select() as HTMLInputElement;
    }
  }

  submitText() {
    console.log('Sent: ' + this.searchStr);
    this.textService.submitText(this.searchStr).subscribe((response: any) => {
      console.log('Received: ' + response);
      this.responseText = response.value;
    });
  }

  public modelChange(str: string): void {
    console.log('triggered')
    if (str.substr(str.length - 1) == ' '){
      this.textService.submitText(str).subscribe((response: any) => {
        console.log('Received: ' + response);
        this.responseText = response.response;
      });
    }
  }

}
