import { Component } from '@angular/core';
import {TextService} from './text.service';

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

  submitText() {
    console.log('Sent: ' + this.searchStr);
    this.textService.submitText(this.searchStr).subscribe((response: any) => {
      console.log('Received: ' + response);
      this.responseText = response.value;
    });
  }

  public modelChange(str: string): void {
    if (str.substr(str.length - 1) == ' '){
      this.textService.submitText(str).subscribe((response: any) => {
        console.log('Received: ' + response);
        this.responseText = response.response;
      });
    }
  }

}
